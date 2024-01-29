import logging
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from orjson import orjson
from pynumaflow.mapper import Messages, Datum, Message

from numalogic.config import PostprocessFactory, RegistryFactory
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

    @staticmethod
    def _calculate_unified_score(
        anomaly_scores: np.ndarray, scoring_function: str
    ) -> Optional[float]:
        if scoring_function == "max":
            return np.max(anomaly_scores)
        if scoring_function == "min":
            return np.min(anomaly_scores)
        if scoring_function == "mean":
            return np.mean(anomaly_scores)
        return None

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
                anomaly_scores = self.compute(
                    model=thresh_artifact.artifact,
                    input_=payload.get_data(),
                    postproc_clf=postproc_clf,
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
                    data=anomaly_scores,
                    header=Header.MODEL_INFERENCE,
                )
                out_payload = OutputPayload(
                    uuid=payload.uuid,
                    config_id=payload.config_id,
                    pipeline_id=payload.pipeline_id,
                    composite_keys=payload.composite_keys,
                    timestamp=payload.end_ts,
                    unified_anomaly=self._calculate_unified_score(
                        anomaly_scores,
                        _conf.unified_scoring_conf.scoring_function,
                    ),
                    data=self._per_feature_score(payload.metrics, anomaly_scores),
                    metadata=payload.metadata,
                )
                _LOGGER.info(
                    "%s - Successfully post-processed, Keys: %s, Scores: %s, Payload: %s "
                    "using strategy: %s",
                    out_payload.uuid,
                    out_payload.composite_keys,
                    out_payload.unified_anomaly,
                    payload,
                    _conf.unified_scoring_conf.scoring_function.value,
                )
                _LOGGER.info("%s-%s", payload.uuid, out_payload)
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
        if scores.shape[0] == 1:
            # TODO improve this to incorporate per feature anomaly insights
            return {_name: scores.item() for _name in feat_names}
        return dict(zip(feat_names, scores))

    @classmethod
    def compute(
        cls, model: artifact_t, input_: NDArray[float], postproc_clf=None, **_
    ) -> NDArray[float]:
        """
        Compute the postprocess function.

        Args:
        -------
        model: Model instance
        input_: Input data
        kwargs: Additional arguments

        Returns
        -------
        Output data
        """
        _start_time = time.perf_counter()
        try:
            score = np.mean(input_, axis=0, keepdims=True)
            y_score = model.score_samples(score).astype(np.float32)
        except Exception as err:
            raise RuntimeError("Threshold model scoring failed") from err
        try:
            score = postproc_clf.transform(y_score)
        except Exception as err:
            raise RuntimeError("Postprocess failed") from err
        _LOGGER.debug(
            "Time taken in postprocess compute: %.4f sec", time.perf_counter() - _start_time
        )
        return score.reshape(-1)
