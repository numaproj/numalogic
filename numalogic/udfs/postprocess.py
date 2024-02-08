import logging
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from orjson import orjson
from pynumaflow.mapper import Messages, Datum, Message

from numalogic.config import PostprocessFactory, RegistryFactory, ScoreConf, AggregatorFactory
from numalogic.tools.aggregators import aggregate_window, aggregate_features
from numalogic.registry import LocalLRUCache
from numalogic.tools.types import redis_client_t, artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf
from numalogic.udfs._metrics import (
    MODEL_STATUS_COUNTER,
    RUNTIME_ERROR_COUNTER,
    MSG_PROCESSED_COUNTER,
    MSG_IN_COUNTER,
    UDF_TIME,
    _increment_counter,
)
from numalogic.udfs.entities import StreamPayload, Header, Status, TrainerPayload, OutputPayload
from numalogic.udfs.tools import _load_artifact

# TODO: move to config
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", "10000"))
LOAD_LATEST = os.getenv("LOAD_LATEST", "false").lower() == "true"

_LOGGER = logging.getLogger(__name__)


class PostprocessUDF(NumalogicUDF):
    """
    Postprocess UDF for Numalogic.

    Args:
        r_client: Redis client
        pl_conf: PipelineConf object
    """

    __slots__ = ("registry_conf", "model_registry", "postproc_factory")

    def __init__(
        self,
        r_client: redis_client_t,
        pl_conf: Optional[PipelineConf] = None,
    ):
        super().__init__(pl_conf=pl_conf, _vtx="postprocess")
        self.registry_conf = self.pl_conf.registry_conf
        model_registry_cls = RegistryFactory.get_cls(self.registry_conf.name)
        self.model_registry = model_registry_cls(
            client=r_client,
            cache_registry=LocalLRUCache(
                ttl=LOCAL_CACHE_TTL,
                cachesize=LOCAL_CACHE_SIZE,
                jitter_sec=self.registry_conf.jitter_conf.jitter_sec,
                jitter_steps_sec=self.registry_conf.jitter_conf.jitter_steps_sec,
            ),
        )
        self.postproc_factory = PostprocessFactory()

    @UDF_TIME.time()
    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        The postprocess function here receives data from the previous udf.

        Args:
        -------
        keys: List of keys
        datum: Datum object.

        Returns
        -------
        Messages instance

        """
        _start_time = time.perf_counter()
        messages = Messages()

        # Construct payload object
        payload = StreamPayload(**orjson.loads(datum.value))
        _metric_label_values = (
            payload.composite_keys,
            self._vtx,
            ":".join(payload.composite_keys),
            payload.config_id,
            payload.pipeline_id,
        )

        _increment_counter(
            counter=MSG_IN_COUNTER,
            labels=_metric_label_values,
        )

        # load configs
        _stream_conf = self.get_stream_conf(payload.config_id)
        _conf = _stream_conf.ml_pipelines[payload.pipeline_id]
        thresh_cfg = _conf.numalogic_conf.threshold
        postprocess_cfg = _conf.numalogic_conf.postprocess

        # load artifact
        thresh_artifact, payload = _load_artifact(
            skeys=[_ckey for _, _ckey in zip(_stream_conf.composite_keys, payload.composite_keys)],
            dkeys=[payload.pipeline_id, thresh_cfg.name],
            payload=payload,
            model_registry=self.model_registry,
            load_latest=LOAD_LATEST,
            vertex=self._vtx,
        )
        postproc_clf = self.postproc_factory.get_instance(postprocess_cfg)

        if thresh_artifact is None:
            payload = replace(
                payload, status=Status.ARTIFACT_NOT_FOUND, header=Header.TRAIN_REQUEST
            )
            _increment_counter(
                MODEL_STATUS_COUNTER,
                labels=(payload.status.value, *_metric_label_values),
            )

        #  Postprocess payload
        if payload.status in (Status.ARTIFACT_FOUND, Status.ARTIFACT_STALE) and thresh_artifact:
            try:
                feature_scores = self.compute(
                    model=thresh_artifact.artifact,
                    input_=payload.get_data(),
                    score_conf=_conf.numalogic_conf.score,
                    postproc_clf=postproc_clf,
                )  # (nfeat,)

                unified_score = self.compute_unified_score(
                    feature_scores, score_conf=_conf.numalogic_conf.score
                )

            except RuntimeError:
                _increment_counter(RUNTIME_ERROR_COUNTER, _metric_label_values)
                _LOGGER.exception(
                    "%s - Runtime postprocess error! Keys: %s, Metric: %s",
                    payload.uuid,
                    payload.composite_keys,
                    payload.metrics,
                )
                payload = replace(payload, status=Status.RUNTIME_ERROR, header=Header.TRAIN_REQUEST)
            else:
                payload = replace(
                    payload,
                    data=feature_scores,
                    header=Header.MODEL_INFERENCE,
                )
                out_payload = OutputPayload(
                    uuid=payload.uuid,
                    config_id=payload.config_id,
                    pipeline_id=payload.pipeline_id,
                    composite_keys=payload.composite_keys,
                    timestamp=payload.end_ts,
                    unified_anomaly=unified_score,
                    data=self._per_feature_score(payload.metrics, feature_scores),
                    metadata=payload.metadata,
                )
                _LOGGER.info(
                    "%s - Successfully post-processed, Keys: %s, Score: %s, Feature Scores: %s",
                    out_payload.uuid,
                    out_payload.composite_keys,
                    out_payload.unified_anomaly,
                    feature_scores.tolist(),
                )
                messages.append(Message(keys=keys, value=out_payload.to_json(), tags=["output"]))

        # Forward payload if a training request is tagged
        if payload.header == Header.TRAIN_REQUEST or payload.status == Status.ARTIFACT_STALE:
            ckeys = [_ckey for _, _ckey in zip(_stream_conf.composite_keys, payload.composite_keys)]
            train_payload = TrainerPayload(
                uuid=payload.uuid,
                composite_keys=ckeys,
                metrics=payload.metrics,
                config_id=payload.config_id,
                pipeline_id=payload.pipeline_id,
            )
            messages.append(Message(keys=keys, value=train_payload.to_json(), tags=["train"]))
        _LOGGER.debug(
            "%s -  Time taken in postprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        _increment_counter(
            MSG_PROCESSED_COUNTER,
            labels=_metric_label_values,
        )
        return messages

    @staticmethod
    def _per_feature_score(feat_names: list[str], scores: NDArray[float]) -> dict[str, float]:
        if (scores_len := len(scores)) != len(feat_names):
            _LOGGER.debug(
                "Scores length: %s does not match feat_names: %s",
                scores_len,
                feat_names,
            )
            return {}
        return dict(zip(feat_names, scores))

    @classmethod
    def compute(
        cls,
        model: artifact_t,
        input_: NDArray[float],
        score_conf: Optional[ScoreConf] = None,
        postproc_clf=None,
        **_,
    ) -> NDArray[float]:
        """
        Compute thresholding, window aggregation followed by postprocess.

        Args:
        -------
        model: Model instance
        input_: Input data (Shape: seq_len x n_features)
        kwargs: Additional arguments

        Returns
        -------
        Output data of shape (n_features, )

        Raises
        ------
            RuntimeError: If threshold model or postproc function fails
        """
        if score_conf is None:
            raise RuntimeError("Score config not provided!")

        scores = cls.compute_threshold(model, input_)  # (seqlen x nfeat)
        win_scores = cls.compute_feature_scores(scores, score_conf)
        if postproc_clf:
            win_scores = cls.compute_postprocess(postproc_clf, win_scores)  # (seqlen x nfeat)
        return win_scores  # (nfeat, )

    @classmethod
    def compute_threshold(cls, model: artifact_t, input_: NDArray[float]) -> NDArray[float]:
        try:
            scores = model.score_samples(input_).astype(np.float32)
        except Exception as err:
            raise RuntimeError("Threshold model scoring failed") from err
        return scores

    @classmethod
    def compute_postprocess(cls, tx: artifact_t, input_: NDArray[float]):
        try:
            score = tx.transform(input_)
        except Exception as err:
            raise RuntimeError("Postprocess failed") from err
        return score

    @classmethod
    def compute_feature_scores(
        cls, scores: NDArray[float], score_conf: ScoreConf
    ) -> NDArray[float]:
        return aggregate_window(
            scores,
            agg_func=AggregatorFactory.get_func(score_conf.window_agg.method),
            **score_conf.window_agg.conf,
        )

    @classmethod
    def compute_unified_score(cls, scores: NDArray[float], score_conf: ScoreConf) -> float:
        return aggregate_features(
            scores.reshape(1, -1),
            agg_func=AggregatorFactory.get_func(score_conf.feature_agg.method),
            **score_conf.feature_agg.conf,
        ).item()
