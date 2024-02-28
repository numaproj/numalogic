import logging
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from orjson import orjson
from pynumaflow.mapper import Messages, Datum, Message

from numalogic.config import (
    PostprocessFactory,
    RegistryFactory,
    ScoreConf,
    AggregatorFactory,
    AggregatorConf,
)
from numalogic.models.threshold import SigmoidThreshold
from numalogic.registry import LocalLRUCache
from numalogic.tools.aggregators import aggregate_window, aggregate_features
from numalogic.tools.types import redis_client_t, artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf, MLPipelineConf
from numalogic.udfs._metrics import (
    RUNTIME_ERROR_COUNTER,
    MSG_PROCESSED_COUNTER,
    MSG_IN_COUNTER,
    UDF_TIME,
    _increment_counter,
)
from numalogic.udfs.entities import StreamPayload, Header, Status, OutputPayload
from numalogic.udfs.tools import _load_artifact, get_trainer_message

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
            return Messages(get_trainer_message(keys, _stream_conf, payload, *_metric_label_values))

        if payload.header == Header.STATIC_INFERENCE:
            _LOGGER.warning("Static inference not supported in postprocess yet")

        #  Postprocess payload
        try:
            # Compute anomaly scores per feature
            a_features = self.compute(
                model=thresh_artifact.artifact,
                input_=payload.get_data(),
                score_conf=_conf.numalogic_conf.score,
                postproc_clf=postproc_clf,
            )  # (nfeat,)

            # Compute unified score
            a_unified = self.compute_unified_score(
                a_features, feat_agg_conf=_conf.numalogic_conf.score.feature_agg
            )

            # Calculate adjusted unified score
            a_adjusted, y_unified, y_features = self._adjust_score(_conf, a_unified, payload)

        except RuntimeError:
            _increment_counter(RUNTIME_ERROR_COUNTER, _metric_label_values)
            _LOGGER.exception(
                "%s - Runtime postprocess error! Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return Messages(get_trainer_message(keys, _stream_conf, payload, *_metric_label_values))

        payload = replace(
            payload,
            data=a_features,
            header=Header.MODEL_INFERENCE,
        )
        out_data = self._additional_scores(
            payload.metrics, a_features, a_unified, y_features, y_unified
        )
        out_payload = OutputPayload(
            uuid=payload.uuid,
            config_id=payload.config_id,
            pipeline_id=payload.pipeline_id,
            composite_keys=payload.composite_keys,
            timestamp=payload.end_ts,
            unified_anomaly=a_adjusted,
            data=out_data,
            metadata=payload.metadata,
        )
        _LOGGER.info(
            "%s - Successfully post-processed, Keys: %s, Score: %s, "
            "Model Score: %s, Static Score: %s, Feature "
            "Scores: %s",
            out_payload.uuid,
            out_payload.composite_keys,
            out_payload.unified_anomaly,
            a_unified,
            y_unified,
            a_features.tolist(),
        )

        _increment_counter(
            MSG_PROCESSED_COUNTER,
            labels=_metric_label_values,
        )
        _LOGGER.debug(
            "%s -  Time taken in postprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        return Messages(Message(keys=keys, value=out_payload.to_json(), tags=["output"]))

    def _adjust_score(
        self, mlpl_conf: MLPipelineConf, a_unified: float, payload: StreamPayload
    ) -> tuple[float, Optional[float], Optional[NDArray[float]]]:
        """
        Adjust the unified score using static threshold scores.

        Args:
        -------
            mlpl_conf: MLPipelineConf object
            a_unified: Unified anomaly score
            payload: StreamPayload object

        Returns
        -------
            A tuple consisting of
                Adjusted unified score,
                Unified static threshold score,
                Static threshold scores per feature
        """
        adjust_conf = mlpl_conf.numalogic_conf.score.adjust
        if adjust_conf:
            # Compute static threshold scores
            y_features = self.compute_static_threshold(
                input_=payload.get_data(original=True, metrics=list(adjust_conf.upper_limits)),
                score_conf=mlpl_conf.numalogic_conf.score,
            )  # (nfeat,)

            # Compute unified static threshold score
            y_unified = self.compute_unified_score(
                y_features, feat_agg_conf=adjust_conf.feature_agg
            )

            # Compute adjusted unified score
            a_adjusted = self.compute_adjusted_score(a_unified, y_unified)
            _LOGGER.debug("y_unified: %s, y_features: %s", y_unified, y_features)
            return a_adjusted, y_unified, y_features
        return a_unified, None, None

    def _additional_scores(
        self,
        feat_names: list[str],
        a_features: NDArray[float],
        a_unified: float,
        y_features: Optional[NDArray[float]] = None,
        y_unified: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Get additional scores.

        Args:
        -------
            feat_names: Feature names
            a_features: ML model anomaly scores per feature
            a_unified: ML model unified anomaly score
            y_features: Static threshold scores per feature
            y_unified: Static threshold unified score

        Returns
        -------
            Dictionary with additional output scores
        """
        data = self._per_feature_score(feat_names, a_features)
        data["namespace_app_rollouts_ML"] = a_unified
        if y_unified is not None:
            data["namespace_app_rollouts_ST"] = y_unified
        if y_features is not None:
            data |= self._per_feature_score([f"{f}_ST" for f in feat_names], y_features)
        return data

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
        postproc_tx=None,
        score_conf: Optional[ScoreConf] = None,
        **_,
    ) -> NDArray[float]:
        """
        Compute thresholding, window aggregation followed by postprocess.

        Args:
        -------
        model: Threshold model instance
        input_: Input data (Shape: seq_len x n_features)
        postproc_tx: Postprocess transform
        score_conf: Score Config

        Returns
        -------
        Output data of shape (n_features, )

        Raises
        ------
            RuntimeError: If threshold model or postproc function fails
        """
        if score_conf is None:
            _LOGGER.warning("Score config not provided, using default values")
            score_conf = ScoreConf()

        scores = cls.compute_threshold(model, input_)  # (seqlen x nfeat)
        win_scores = cls.compute_feature_scores(
            scores, win_agg_conf=score_conf.window_agg
        )  # (seqlen x nfeat)
        if postproc_tx:
            win_scores = cls.compute_postprocess(postproc_tx, win_scores)  # (seqlen x nfeat)
        return win_scores  # (nfeat, )

    @classmethod
    def compute_threshold(cls, model: artifact_t, input_: NDArray[float]) -> NDArray[float]:
        """
        Compute raw anomaly scores using the threshold model.

        Args:
        -------
            model: Threshold model
            input_: Input data (Shape: seq_len x n_features)

        Returns
        -------
            Raw anomaly scores of shape (seq_len x n_features)

        Raises
        ------
            RuntimeError: If threshold model scoring fails
        """
        try:
            scores = model.score_samples(input_).astype(np.float32)
        except Exception as err:
            raise RuntimeError("Threshold model scoring failed") from err
        return scores

    @classmethod
    def compute_postprocess(cls, tx: artifact_t, input_: NDArray[float]) -> NDArray[float]:
        """
        Postprocess the input scores using the postprocess transform.

        Args:
        -------
            tx: Postprocess transform function
            input_: Input data (Shape: seq_len x n_features)

        Returns
        -------
            Postprocessed scores of shape (seq_len x n_features)

        Raises
        ------
            RuntimeError: If postprocess fails
        """
        try:
            score = tx.transform(input_)
        except Exception as err:
            raise RuntimeError("Postprocess failed") from err
        return score

    @classmethod
    def compute_feature_scores(
        cls, scores: NDArray[float], win_agg_conf: AggregatorConf
    ) -> NDArray[float]:
        """
        Aggregate scores over the window length.

        Args:
        -------
            scores: anomaly scores (Shape: seq_len x n_features)
            win_agg_conf: Window aggregator Config

        Returns
        -------
            Aggregated scores of shape (n_features,)
        """
        return aggregate_window(
            scores,
            agg_func=AggregatorFactory.get_func(win_agg_conf.method),
            **win_agg_conf.conf,
        )

    @classmethod
    def compute_unified_score(cls, scores: NDArray[float], feat_agg_conf: AggregatorConf) -> float:
        """
        Aggregate scores over the features to get a unified score.

        Args:
        -------
            scores: anomaly scores (Shape: n_features, )
            feat_agg_conf: Feature aggregator Config

        Returns
        -------
            Unified score (float)
        """
        return aggregate_features(
            scores.reshape(1, -1),
            agg_func=AggregatorFactory.get_func(feat_agg_conf.method),
            **feat_agg_conf.conf,
        ).item()

    @classmethod
    def compute_static_threshold(
        cls, input_: NDArray[float], score_conf: ScoreConf
    ) -> NDArray[float]:
        """
        Compute static thresholding over the raw input features.

        Args:
        -------
            input_: Input data (Shape: seq_len x n_features)
            score_conf: Score Config

        Returns
        -------
            Aggregated threshold scores of shape (n_features,)
        """
        clf = SigmoidThreshold(*score_conf.adjust.upper_limits.values())
        y_features = clf.score_samples(input_)
        return cls.compute_feature_scores(y_features, score_conf.adjust.window_agg)

    @classmethod
    def compute_adjusted_score(cls, a_unified: float, y_unified: float) -> float:
        """
        Compute adjusted unified score using static threshold scores.

        Args:
        -------
            a_unified: Unified anomaly score
            y_unified: Unified static threshold score

        Returns
        -------
            Adjusted unified score (float)
        """
        return max(a_unified, y_unified)
