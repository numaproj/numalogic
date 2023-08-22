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

from numalogic.config import NumalogicConf
from numalogic.registry import RedisRegistry, LocalLRUCache, ArtifactData
from numalogic.tools.exceptions import RedisRegistryError, ModelKeyNotFound
from numalogic.tools.types import artifact_t, redis_client_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs.entities import StreamPayload, Header, Status

_LOGGER = logging.getLogger(__name__)
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
RETRAIN_FREQ_HR = int(os.getenv("RETRAIN_FREQ_HR", "24"))


class InferenceUDF(NumalogicUDF):

    def __init__(self, r_client: redis_client_t, numalogic_conf: Optional[NumalogicConf] = None):
        super().__init__(is_async=False)
        self.model_registry = RedisRegistry(
            client=r_client, cache_registry=LocalLRUCache(ttl=LOCAL_CACHE_TTL)
        )
        self.numalogic_conf = numalogic_conf or NumalogicConf()

    def exec(self, keys: list[str], datum: Datum) -> Messages:
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
        if payload.header is Header.TRAIN_REQUEST:
            return Messages(Message(keys=keys, value=payload.to_json()))

        artifact_data = self.load_artifact(keys, payload)
        # Send training request if artifact loading is not successful
        if not artifact_data:
            payload = replace(
                payload, status=Status.ARTIFACT_NOT_FOUND, header=Header.TRAIN_REQUEST
            )
            Messages(Message(keys=keys, value=payload.to_json()))

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

    def load_artifact(self, keys: list[str], payload: StreamPayload) -> Optional[ArtifactData]:
        try:
            artifact_data = self.model_registry.load(
                skeys=keys,
                dkeys=[self.numalogic_conf.model.name],
            )
        except ModelKeyNotFound:
            _LOGGER.warning(
                "%s - Model key not found for Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
                exc_info=True,
            )
            return None
        except RedisRegistryError:
            _LOGGER.exception(
                "%s - Error while fetching inference artifact, Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return None
        _LOGGER.info(
            "%s - Loaded artifact data from %s",
            payload.uuid,
            artifact_data.extras.get("source"),
        )
        return artifact_data

    def is_model_stale(self, artifact_data: ArtifactData, payload: StreamPayload) -> bool:
        if (
            self.model_registry.is_artifact_stale(artifact_data, int(RETRAIN_FREQ_HR))
            and artifact_data.extras.get("source") == "registry"
        ):
            _LOGGER.info(
                "%s - Inference artifact found is stale, Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return True
        return False

    def compute(self, model: artifact_t, input_: npt.NDArray[float]) -> npt.NDArray[float]:
        x = torch.from_numpy(input_).unsqueeze(0)
        model.eval()
        try:
            with torch.no_grad():
                _, out = model.forward(x)
            recon_err = model.criterion(out, x, reduction="none")
        except Exception as err:
            raise RuntimeError("Model forward pass failed!") from err
        return np.ascontiguousarray(recon_err).squeeze(0)
