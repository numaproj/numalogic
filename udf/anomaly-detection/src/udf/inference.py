import os
import time
from dataclasses import replace
from typing import List

import numpy as np
import orjson
import torch
from numalogic.config import ModelFactory, NumalogicConf
from numalogic.registry import RedisRegistry, ArtifactData, LocalLRUCache
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.tools.types import artifact_t, nn_model_t, state_dict_t
from pynumaflow.function import Datum, Messages, Message

from src import get_logger
from src.connectors.sentinel import get_redis_client_from_conf
from src.entities import Status, Header
from src.entities import StreamPayload
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", 3600))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", 10000))


class Inference:
    def __init__(self):
        local_cache = LocalLRUCache(cachesize=LOCAL_CACHE_SIZE, ttl=LOCAL_CACHE_TTL)
        self.model_registry = RedisRegistry(
            client=get_redis_client_from_conf(master_node=False), cache_registry=local_cache
        )
        self.config_manager = ConfigManager()
        self.static_response = ([[]], Status.ARTIFACT_NOT_FOUND, Header.STATIC_INFERENCE, -1)
        self.model_factory = ModelFactory()

    @staticmethod
    def forward_pass(payload: StreamPayload, model: nn_model_t) -> np.ndarray:
        x = torch.from_numpy(payload.get_data()).unsqueeze(0)
        model.eval()
        try:
            with torch.no_grad():
                _, out = model.forward(x)
            recon_err = model.criterion(out, x, reduction="none")
        except Exception as err:
            raise RuntimeError(f"Model forward pass failed!, Error: {err}") from err
        return np.ascontiguousarray(recon_err).squeeze(0)

    def inference(
        self, keys: List[str], payload: StreamPayload
    ) -> (np.ndarray, Status, Header, int):
        # Check if metric needs static inference
        if payload.header == Header.STATIC_INFERENCE:
            _LOGGER.debug(
                "%s - Models not found in the previous steps, forwarding for static thresholding. Keys: %s, Metrics: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return self.static_response

        # Load config
        retrain_config = self.config_manager.get_retrain_config(config_id=payload.config_id)
        numalogic_conf = self.config_manager.get_numalogic_config(config_id=payload.config_id)

        # Load inference artifact
        try:
            artifact_data = self.model_registry.load(
                skeys=keys,
                dkeys=[numalogic_conf.model.name],
            )
        except RedisRegistryError as err:
            _LOGGER.exception(
                "%s - Error while fetching inference artifact, Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
                err,
            )
            return self.static_response
        except Exception as ex:
            _LOGGER.exception(
                "%s - Unhandled exception while fetching  inference artifact, Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
                ex,
            )
            return self.static_response

        # Check if artifact is found
        if not artifact_data:
            _LOGGER.info(
                "%s - Inference artifact not found, forwarding for static thresholding. Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return self.static_response

        # Check if artifact is stale
        header = Header.MODEL_INFERENCE

        _LOGGER.info(
            "%s - Loaded artifact data from %s",
            payload.uuid,
            artifact_data.extras.get("source"),
        )
        if (
            RedisRegistry.is_artifact_stale(artifact_data, int(retrain_config.retrain_freq_hr))
            and artifact_data.extras.get("source") == "registry"
        ):
            _LOGGER.info(
                "%s - Inference artifact found is stale, Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            header = Header.MODEL_STALE

        # Generate predictions
        try:
            model = self.load_state_dict(numalogic_conf, artifact_data.artifact)
            x_inferred = self.forward_pass(payload, model)
        except RuntimeError as err:
            _LOGGER.error(
                "%s - Failed to infer, forwarding for static thresholding. Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
                err,
            )
            return [[]], Status.RUNTIME_ERROR, Header.STATIC_INFERENCE, -1
        _LOGGER.info("%s - Successfully inferred: Metric: %s", payload.uuid, payload.metrics)
        return x_inferred, Status.INFERRED, header, int(artifact_data.extras.get("version"))

    def load_state_dict(
        self, numalogic_conf: NumalogicConf, state_dict: state_dict_t
    ) -> nn_model_t:
        model = self.model_factory.get_instance(numalogic_conf.model)
        try:
            model.load_state_dict(state_dict)
        except TypeError as type_err:
            raise RuntimeError(f"Failed to load state dict, Error: {type_err}") from type_err
        return model

    def run(self, keys: List[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()

        # Construct payload object
        payload = StreamPayload(**orjson.loads(datum.value))

        _LOGGER.info(
            "%s - Received Msg: { CompositeKeys: %s, Metrics: %s }",
            payload.uuid,
            payload.composite_keys,
            payload.metrics,
        )

        # Perform inference
        x_inferred, status, header, version = self.inference(keys, payload)

        payload = replace(
            payload,
            status=status,
            header=header,
            data=x_inferred,
            metadata={"model_version": version, **payload.metadata},
        )

        _LOGGER.info(
            "%s - Sending Msg: { CompositeKeys: %s, Metrics: %s }",
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
