import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import torch
from numpy import typing as npt
from orjson import orjson
from pynumaflow.mapper import Messages, Datum, Message

from numalogic.config import RegistryFactory
from numalogic.registry import LocalLRUCache, ArtifactData
from numalogic.tools.types import artifact_t, redis_client_t
from numalogic.udfs._base import NumalogicUDF
from numalogic.udfs._config import PipelineConf

from numalogic.udfs._logger import configure_logger, log_data_payload_values
from numalogic.udfs._metrics_utility import _increment_counter
from numalogic.udfs.entities import StreamPayload, Status
from numalogic.udfs.tools import (
    _load_artifact,
    _update_gauge_metric,
    get_trainer_message,
    get_static_thresh_message,
)

_struct_log = configure_logger()

# TODO: move to config
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", "10000"))
LOAD_LATEST = os.getenv("LOAD_LATEST", "false").lower() == "true"
METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))


class InferenceUDF(NumalogicUDF):
    """
    Inference UDF for Numalogic.

    Args:
        r_client: Redis client
        pl_conf: Stream configuration per config ID
    """

    __slots__ = ("registry_conf", "model_registry")

    def __init__(self, r_client: redis_client_t, pl_conf: Optional[PipelineConf] = None):
        super().__init__(is_async=False, pl_conf=pl_conf, _vtx="inference")
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

    @classmethod
    def compute(cls, model: artifact_t, input_: npt.NDArray[float], **_) -> npt.NDArray[float]:
        """
        Perform inference on the input data.

        Args:
            model: Model artifact
            input_: Input data

        Returns
        -------
            Reconstruction error

        Raises
        ------
            RuntimeError: If model forward pass fails
        """
        x = torch.from_numpy(input_).unsqueeze(0)
        model.eval()
        try:
            with torch.no_grad():
                recon_err = model.get_reconstruction_loss(x, reduction="none")
        except Exception as err:
            raise RuntimeError("Model forward pass failed!") from err
        return np.ascontiguousarray(recon_err).squeeze(0)

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        Perform inference on the input data.

        Args:
            keys: List of keys
            datum: Datum object

        Returns
        -------
            Messages instance
        """
        _start_time = time.perf_counter()
        logger = _struct_log.bind(udf_vertex=self._vtx)

        # Construct payload object
        json_data_payload = orjson.loads(datum.value)
        payload = StreamPayload(**json_data_payload)
        _metric_label_values = {
            "source": payload.metadata["numalogic_opex_tags"]["source"],
            "vertex": self._vtx,
            "composite_key": ":".join(payload.composite_keys),
            "config_id": payload.config_id,
            "pipeline_id": payload.pipeline_id,
        }
        _increment_counter(
            counter="MSG_IN_COUNTER",
            labels=_metric_label_values,
            is_enabled=METRICS_ENABLED,
        )

        _stream_conf = self.get_stream_conf(payload.config_id)
        _conf = _stream_conf.ml_pipelines[payload.pipeline_id]

        logger = log_data_payload_values(logger, json_data_payload)

        artifact_data, payload = _load_artifact(
            skeys=[_ckey for _, _ckey in zip(_stream_conf.composite_keys, payload.composite_keys)],
            dkeys=[payload.pipeline_id, _conf.numalogic_conf.model.name],
            payload=payload,
            model_registry=self.model_registry,
            load_latest=LOAD_LATEST,
            vertex=self._vtx,
        )

        # Send training request if artifact loading is not successful
        if not artifact_data:
            msgs = Messages(get_trainer_message(keys, _stream_conf, payload))
            if _conf.numalogic_conf.score.adjust:
                msgs.append(get_static_thresh_message(keys, payload))
            logger.exception("Artifact model not loaded!")
            return msgs

        # Perform inference
        try:
            x_inferred = self.compute(artifact_data.artifact, payload.get_data())
            _update_gauge_metric(x_inferred, payload.metrics, _metric_label_values)
        except RuntimeError:
            _increment_counter(
                counter="RUNTIME_ERROR_COUNTER",
                labels=_metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            logger.exception(
                "Runtime inference error!",
                keys=payload.composite_keys,
                metrics=payload.metrics,
            )
            # Send training request if inference fails
            msgs = Messages(get_trainer_message(keys, _stream_conf, payload))
            if _conf.numalogic_conf.score.adjust:
                msgs.append(get_static_thresh_message(keys, payload))
            return msgs

        msgs = Messages()
        status = (
            Status.ARTIFACT_STALE
            if self.is_model_stale(artifact_data, payload)
            else Status.ARTIFACT_FOUND
        )
        payload = replace(
            payload,
            data=x_inferred,
            status=status,
            metadata={
                "model_version": int(artifact_data.extras.get("version")),
                **payload.metadata,
            },
        )
        # Send trainer message if artifact is stale
        if status == Status.ARTIFACT_STALE:
            logger.info("Inference artifact found is stale")
            msgs.append(get_trainer_message(keys, _stream_conf, payload, **_metric_label_values))

        _increment_counter(
            counter="MSG_PROCESSED_COUNTER",
            labels=_metric_label_values,
            is_enabled=METRICS_ENABLED,
        )
        msgs.append(Message(keys=keys, value=payload.to_json(), tags=["postprocess"]))

        logger.info(
            "Successfully inferred!",
            keys=payload.composite_keys,
            metrics=payload.metrics,
            execution_time_ms=round((time.perf_counter() - _start_time) * 1000, 4),
        )
        return msgs

    def is_model_stale(self, artifact_data: ArtifactData, payload: StreamPayload) -> bool:
        """
        Check if the inference artifact is stale.

        Args:
            artifact_data: ArtifactData instance
            payload: StreamPayload object

        Returns
        -------
            True if artifact is stale, False otherwise
        """
        _conf = self.get_ml_pipeline_conf(
            config_id=payload.config_id, pipeline_id=payload.pipeline_id
        )
        if artifact_data.extras.get(
            "source", "registry"
        ) == "registry" and self.model_registry.is_artifact_stale(
            artifact_data, _conf.numalogic_conf.trainer.retrain_freq_hr
        ):
            return True
        return False
