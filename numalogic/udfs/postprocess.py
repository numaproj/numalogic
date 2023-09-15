import logging
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from orjson import orjson
from pynumaflow.function import Messages, Datum, Message

from numalogic.config import PostprocessFactory, RegistryFactory
from numalogic.registry import LocalLRUCache
from numalogic.tools.exceptions import ConfigNotFoundError
from numalogic.tools.types import redis_client_t, artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import StreamConf, PipelineConf
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

    def __init__(
        self,
        r_client: redis_client_t,
        pl_conf: Optional[PipelineConf] = None,
    ):
        super().__init__()
        model_registry_cls = RegistryFactory.get_cls("RedisRegistry")
        self.model_registry = model_registry_cls(
            client=r_client, cache_registry=LocalLRUCache(ttl=LOCAL_CACHE_TTL)
        )
        self.pl_conf = pl_conf or PipelineConf()
        self.postproc_factory = PostprocessFactory()

    def register_conf(self, config_id: str, conf: StreamConf) -> None:
        """
        Register config with the UDF.

        Args:
            config_id: Config ID
            conf: StreamConf object
        """
        self.pl_conf.stream_confs[config_id] = conf

    def get_conf(self, config_id: str) -> StreamConf:
        try:
            return self.pl_conf.stream_confs[config_id]
        except KeyError as err:
            raise ConfigNotFoundError(f"Config with ID {config_id} not found!") from err

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

        # load configs
        thresh_cfg = self.get_conf(payload.config_id).numalogic_conf.threshold
        postprocess_cfg = self.get_conf(payload.config_id).numalogic_conf.postprocess

        # load artifact
        thresh_artifact, payload = _load_artifact(
            skeys=keys,
            dkeys=[thresh_cfg.name],
            payload=payload,
            model_registry=self.model_registry,
            load_latest=LOAD_LATEST,
        )
        postproc_clf = self.postproc_factory.get_instance(postprocess_cfg)

        if thresh_artifact is None:
            payload = replace(
                payload, status=Status.ARTIFACT_NOT_FOUND, header=Header.TRAIN_REQUEST
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
                    composite_keys=payload.composite_keys,
                    timestamp=payload.end_ts,
                    unified_anomaly=np.max(anomaly_scores),
                    data={
                        _metric: _score for _metric, _score in zip(payload.metrics, anomaly_scores)
                    },
                    # TODO: add model version, & emit as ML metrics
                    metadata=payload.metadata,
                )
                _LOGGER.info(
                    "%s - Successfully post-processed, Keys: %s, Scores: %s",
                    out_payload.uuid,
                    out_payload.composite_keys,
                    out_payload.data,
                )
                messages.append(Message(keys=keys, value=out_payload.to_json(), tags=["output"]))

        # Forward payload if a training request is tagged
        if payload.header == Header.TRAIN_REQUEST or payload.status == Status.ARTIFACT_STALE:
            train_payload = TrainerPayload(
                uuid=payload.uuid,
                composite_keys=keys,
                metrics=payload.metrics,
                config_id=payload.config_id,
            )
            messages.append(Message(keys=keys, value=train_payload.to_json(), tags=["train"]))
        _LOGGER.debug(
            "%s -  Time taken in postprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        return messages

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
            y_score = model.score_samples(input_).astype(np.float32)
        except Exception as err:
            raise RuntimeError("Threshold model scoring failed") from err
        try:
            win_score = np.mean(y_score, axis=0)
            score = postproc_clf.transform(win_score)
        except Exception as err:
            raise RuntimeError("Postprocess failed") from err
        _LOGGER.debug(
            "Time taken in postprocess compute: %.4f sec", time.perf_counter() - _start_time
        )
        return score
