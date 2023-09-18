import logging
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import torch
from numpy import typing as npt
from orjson import orjson
from pynumaflow.function import Messages, Datum, Message

from numalogic.config import RegistryFactory
from numalogic.registry import LocalLRUCache, ArtifactData
from numalogic.tools.exceptions import ConfigNotFoundError
from numalogic.tools.types import artifact_t, redis_client_t
from numalogic.udfs._base import NumalogicUDF
from numalogic.udfs._config import StreamConf, PipelineConf
from numalogic.udfs.entities import StreamPayload, Header, Status
from numalogic.udfs.tools import _load_artifact

_LOGGER = logging.getLogger(__name__)

# TODO: move to config
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", "10000"))
LOAD_LATEST = os.getenv("LOAD_LATEST", "false").lower() == "true"


class InferenceUDF(NumalogicUDF):
    """
    Inference UDF for Numalogic.

    Args:
        r_client: Redis client
        pl_conf: Stream configuration per config ID
    """

    def __init__(self, r_client: redis_client_t, pl_conf: Optional[PipelineConf] = None):
        super().__init__(is_async=False)
        model_registry_cls = RegistryFactory.get_cls("RedisRegistry")
        self.model_registry = model_registry_cls(
            client=r_client,
            cache_registry=LocalLRUCache(ttl=LOCAL_CACHE_TTL, cachesize=LOCAL_CACHE_SIZE),
        )
        self.pl_conf = pl_conf or PipelineConf()

    # TODO: remove, and have an update config method
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
                _, out = model.forward(x)
                recon_err = model.criterion(out, x, reduction="none")
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

        # Construct payload object
        payload = StreamPayload(**orjson.loads(datum.value))

        _LOGGER.debug(
            "%s - Received Msg: { CompositeKeys: %s, Metrics: %s }",
            payload.uuid,
            payload.composite_keys,
            payload.metrics,
        )

        # Forward payload if a training request is tagged
        if payload.header == Header.TRAIN_REQUEST:
            return Messages(Message(keys=keys, value=payload.to_json()))
        artifact_data, payload = _load_artifact(
            skeys=keys,
            dkeys=[self.get_conf(payload.config_id).numalogic_conf.model.name],
            payload=payload,
            model_registry=self.model_registry,
            load_latest=LOAD_LATEST,
        )

        # TODO: revisit retraining logic
        # Send training request if artifact loading is not successful
        if not artifact_data:
            payload = replace(
                payload, status=Status.ARTIFACT_NOT_FOUND, header=Header.TRAIN_REQUEST
            )
            return Messages(Message(keys=keys, value=payload.to_json()))

        # Perform inference
        try:
            x_inferred = self.compute(artifact_data.artifact, payload.get_data())
        except RuntimeError:
            _LOGGER.exception(
                "%s - Runtime inference error! Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            payload = replace(payload, status=Status.RUNTIME_ERROR, header=Header.TRAIN_REQUEST)
        else:
            status = (
                Status.ARTIFACT_STALE
                if self.is_model_stale(artifact_data, payload)
                else payload.status
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

        _LOGGER.info(
            "%s - Successfully inferred: { CompositeKeys: %s, Metrics: %s }",
            payload.uuid,
            payload.composite_keys,
            payload.metrics,
        )
        _LOGGER.debug(
            "%s - Time taken in inference: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        return Messages(Message(keys=keys, value=payload.to_json()))

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
        _conf = self.get_conf(payload.config_id)
        if (
            self.model_registry.is_artifact_stale(
                artifact_data, _conf.numalogic_conf.trainer.retrain_freq_hr
            )
            and artifact_data.extras.get("source", "registry") == "registry"
        ):
            _LOGGER.info(
                "%s - Inference artifact found is stale, Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return True
        return False
