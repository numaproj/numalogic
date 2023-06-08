import time

import numpy as np
from typing import List

import orjson
from torch.utils.data import DataLoader

from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import RedisRegistry, ArtifactData
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import RedisRegistryError
from pynumaflow.function import Datum, Messages, Message

from src import get_logger
from src.connectors.sentinel import get_redis_client_from_conf
from src.entities import StreamPayload
from src.entities import Status, Header
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)


class Inference:
    def __init__(self):
        self.model_registry = RedisRegistry(client=get_redis_client_from_conf())

    @classmethod
    def _run_inference(
        cls,
        keys: list[str],
        metric: str,
        payload: StreamPayload,
        artifact_data: ArtifactData,
    ) -> np.ndarray:
        model = artifact_data.artifact
        win_size = ConfigManager.get_datastream_config(config_name=keys[0]).window_size
        metric_arr = payload.get_metric_arr(metric=metric).reshape(-1, 1)
        stream_loader = DataLoader(StreamingDataset(metric_arr, win_size))

        trainer = AutoencoderTrainer()
        try:
            recon_err = trainer.predict(model, dataloaders=stream_loader)
        except Exception as err:
            _LOGGER.exception(
                "%s - Runtime error while performing inference: Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                keys,
                metric,
                err,
            )
            raise RuntimeError("Failed to infer") from err
        _LOGGER.info("%s - Successfully inferred: Keys: %s, Metric: %s", payload.uuid, keys, metric)
        return recon_err.numpy().flatten()

    def inference(
        self, keys: List[str], metric: str, payload: StreamPayload
    ) -> (np.ndarray, Status, Header, int):
        static_response = (None, Status.ARTIFACT_NOT_FOUND, Header.STATIC_INFERENCE, -1)
        # Check if metric needs static inference
        if payload.header[metric] == Header.STATIC_INFERENCE:
            _LOGGER.debug(
                "%s - Models not found in the previous steps, forwarding for static thresholding. Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                metric,
            )
            return static_response

        # Load config
        retrain_config = ConfigManager.get_retrain_config(config_name=keys[0], metric_name=metric)
        numalogic_conf = ConfigManager.get_numalogic_config(config_name=keys[0], metric_name=metric)

        # Load inference artifact
        try:
            artifact_data = self.model_registry.load(
                skeys=keys + [metric],
                dkeys=[numalogic_conf.model.name],
            )
        except RedisRegistryError as err:
            _LOGGER.exception(
                "%s - Error while fetching inference artifact, Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                payload.composite_keys,
                metric,
                err,
            )
            return static_response

        # Check if artifact is found
        if not artifact_data:
            _LOGGER.info(
                "%s - Inference artifact not found, forwarding for static thresholding. Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                metric,
            )
            return static_response

        # Check if artifact is stale
        header = Header.MODEL_INFERENCE
        if RedisRegistry.is_artifact_stale(artifact_data, int(retrain_config.retrain_freq_hr)):
            _LOGGER.info(
                "%s - Inference artifact found is stale, Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                metric,
            )
            header = Header.MODEL_STALE

        # Generate predictions
        try:
            x_infered = self._run_inference(keys, metric, payload, artifact_data)
        except RuntimeError:
            _LOGGER.info(
                "%s - Failed to infer, forwarding for static thresholding. Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                metric,
            )
            return None, Status.RUNTIME_ERROR, Header.STATIC_INFERENCE, -1

        return x_infered, Status.INFERRED, header, int(artifact_data.extras.get("version"))

    def run(self, keys: List[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()
        _ = datum.event_time
        _ = datum.watermark

        # Construct payload object
        _in_msg = datum.value.decode("utf-8")
        payload = StreamPayload(**orjson.loads(_in_msg))

        _LOGGER.info("%s - Received Msg: { Keys: %s, Payload: %r }", payload.uuid, keys, payload)

        messages = Messages()
        # Perform inference for each metric
        for metric in payload.metrics:
            x_infered, status, header, version = self.inference(keys, metric, payload)
            payload.set_status(metric=metric, status=status)
            payload.set_header(metric=metric, header=header)
            payload.set_metric_metadata(metric=metric, key="model_version", value=version)

            if x_infered is not None:
                payload.set_metric_data(metric=metric, arr=x_infered)

        messages.append(Message(keys=keys, value=payload.to_json()))
        _LOGGER.info("%s - Sending Msg: { Keys: %s, Payload: %r }", payload.uuid, keys, payload)
        return messages
