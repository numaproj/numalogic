import os
import time

import numpy as np
from typing import List

import orjson
from torch.utils.data import DataLoader

from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import RedisRegistry, ArtifactData, LocalLRUCache
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import RedisRegistryError
from pynumaflow.function import Datum, Messages, Message

from src import get_logger
from src.connectors.sentinel import get_redis_client_from_conf
from src.entities import StreamPayload
from src.entities import Status, Header
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", 3600))


class Inference:
    def __init__(self):
        local_cache = LocalLRUCache(ttl=LOCAL_CACHE_TTL)
        self.model_registry = RedisRegistry(
            client=get_redis_client_from_conf(master_node=False), cache_registry=local_cache
        )

    @classmethod
    def _run_inference(
        cls,
        keys: list[str],
        payload: StreamPayload,
        artifact_data: ArtifactData,
    ) -> np.ndarray:
        model = artifact_data.artifact
        win_size = ConfigManager.get_stream_config(config_id=payload.config_id).window_size
        data_arr = payload.get_data()
        stream_loader = DataLoader(StreamingDataset(data_arr, win_size))

        trainer = AutoencoderTrainer()
        try:
            recon_err = trainer.predict(model, dataloaders=stream_loader)
        except Exception as err:
            _LOGGER.exception(
                "%s - Runtime error while performing inference: Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                keys,
                payload.metrics,
                err,
            )
            raise RuntimeError("Failed to infer") from err
        _LOGGER.info(
            "%s - Successfully inferred: Keys: %s, Metric: %s", payload.uuid, keys, payload.metrics
        )
        return recon_err.numpy()

    def inference(
        self, keys: List[str], payload: StreamPayload
    ) -> (np.ndarray, Status, Header, int):
        static_response = (None, Status.ARTIFACT_NOT_FOUND, Header.STATIC_INFERENCE, -1)
        # Check if metric needs static inference
        if payload.header == Header.STATIC_INFERENCE:
            _LOGGER.debug(
                "%s - Models not found in the previous steps, forwarding for static thresholding. Keys: %s, Metrics: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return static_response

        # Load config
        retrain_config = ConfigManager.get_retrain_config(config_id=payload.config_id)
        numalogic_conf = ConfigManager.get_numalogic_config(config_id=payload.config_id)

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
            return static_response

        except Exception as ex:
            _LOGGER.exception(
                "%s - Unhandled exception while fetching  inference artifact, Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
                ex,
            )
            return static_response

        # Check if artifact is found
        if not artifact_data:
            _LOGGER.info(
                "%s - Inference artifact not found, forwarding for static thresholding. Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return static_response

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
            x_inferred = self._run_inference(keys, payload, artifact_data)
        except RuntimeError:
            _LOGGER.info(
                "%s - Failed to infer, forwarding for static thresholding. Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            return None, Status.RUNTIME_ERROR, Header.STATIC_INFERENCE, -1

        return x_inferred, Status.INFERRED, header, int(artifact_data.extras.get("version"))

    def run(self, keys: List[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()
        _ = datum.event_time
        _ = datum.watermark

        # Construct payload object
        _in_msg = datum.value.decode("utf-8")
        payload = StreamPayload(**orjson.loads(_in_msg))

        _LOGGER.info("%s - Received Msg: { Keys: %s, Payload: %r }", payload.uuid, keys, payload)

        messages = Messages()
        # Perform inference
        x_inferred, status, header, version = self.inference(keys, payload)
        payload.set_status(status=status)
        payload.set_header(header=header)
        payload.set_metadata(key="model_version", value=version)

        if x_inferred is not None:
            payload.set_data(arr=x_inferred)

        messages.append(Message(keys=keys, value=payload.to_json()))
        _LOGGER.info("%s - Sending Msg: { Keys: %s, Payload: %r }", payload.uuid, keys, payload)
        _LOGGER.debug("%s - Time taken in inference: %.4f sec", payload.uuid, time.perf_counter() - _start_time)
        return messages
