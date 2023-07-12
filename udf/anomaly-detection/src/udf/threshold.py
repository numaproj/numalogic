import time

import numpy as np
from typing import List
from orjson import orjson

from numalogic.registry import RedisRegistry
from numalogic.tools.exceptions import RedisRegistryError
from pynumaflow.function import Datum, Messages, Message

from src import get_logger
from src._constants import TRAIN_VTX_KEY, POSTPROC_VTX_KEY
from src.connectors.sentinel import get_redis_client_from_conf
from src.entities import Status, TrainerPayload, Header, StreamPayload
from src.tools import calculate_static_thresh
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)


class Threshold:
    def __init__(self):
        self.model_registry = RedisRegistry(client=get_redis_client_from_conf())

    def threshold(
        self, keys: List[str], metric: str, payload: StreamPayload
    ) -> (np.ndarray, Status, Header, int):
        metric_arr = payload.get_metric_arr(metric=metric)

        # Load config
        static_thresh = ConfigManager.get_static_threshold_config(
            config_name=keys[0], metric_name=metric
        )
        thresh_cfg = ConfigManager.get_threshold_config(config_name=keys[0], metric_name=metric)

        # Check if metric needs static inference
        if payload.header[metric] == Header.STATIC_INFERENCE:
            _LOGGER.info(
                "%s - Sending to trainer and performing static thresholding. Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                metric,
            )
            static_scores = calculate_static_thresh(metric_arr, static_thresh)
            return static_scores, Status.ARTIFACT_NOT_FOUND, Header.STATIC_INFERENCE, -1

        # Load threshold artifact
        try:
            thresh_artifact = self.model_registry.load(
                skeys=keys + [metric],
                dkeys=[thresh_cfg.name],
            )
        except RedisRegistryError as err:
            _LOGGER.exception(
                "%s - Error while fetching threshold artifact, Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                payload.composite_keys,
                metric,
                err,
            )
            static_scores = calculate_static_thresh(metric_arr, static_thresh)
            return static_scores, Status.RUNTIME_ERROR, Header.STATIC_INFERENCE, -1

        # Check if artifact is found
        if not thresh_artifact:
            _LOGGER.info(
                "%s - Threshold artifact not found, performing static thresholding. Keys: %s",
                payload.uuid,
                payload.composite_keys,
            )
            static_scores = calculate_static_thresh(metric_arr, static_thresh)
            return static_scores, Status.ARTIFACT_NOT_FOUND, Header.STATIC_INFERENCE, -1

        # Calculate anomaly score
        recon_err = payload.get_metric_arr(metric=metric)
        thresh_clf = thresh_artifact.artifact
        y_score = thresh_clf.score_samples(recon_err)

        return (
            y_score,
            Status.THRESHOLD,
            payload.header[metric],
            payload.get_metadata(key=metric)["model_version"],
        )

    def run(self, keys: List[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()
        _ = datum.event_time
        _ = datum.watermark

        # Construct payload object
        _in_msg = datum.value.decode("utf-8")
        payload = StreamPayload(**orjson.loads(_in_msg))

        _LOGGER.info("%s - Received Msg: { Keys: %s, Payload: %r }", payload.uuid, keys, payload)

        messages = Messages()

        # Perform threshold for each metric
        for metric in payload.metrics:
            y_score, status, header, version = self.threshold(keys, metric, payload)
            payload.set_status(metric=metric, status=status)
            payload.set_header(metric=metric, header=header)
            payload.set_metric_metadata(metric=metric, key="model_version", value=version)

            if y_score is not None:
                payload.set_metric_data(metric=metric, arr=y_score)

            if y_score is None or header == Header.MODEL_STALE or status == Status.ARTIFACT_NOT_FOUND:
                train_payload = TrainerPayload(
                    uuid=payload.uuid, composite_keys=keys, metric=metric
                )
                _LOGGER.info(
                    "%s - Sending Msg: { Keys: %s, Tags:%s, Payload: %s }",
                    payload.uuid,
                    keys,
                    [TRAIN_VTX_KEY],
                    train_payload,
                )
                messages.append(
                    Message(keys=keys, value=train_payload.to_json(), tags=[TRAIN_VTX_KEY])
                )

        messages.append(Message(keys=keys, value=payload.to_json(), tags=[POSTPROC_VTX_KEY]))
        _LOGGER.info(
            "%s - Sending Msg: { Keys: %s, Tags:%s, Payload: %r }",
            payload.uuid,
            keys,
            [POSTPROC_VTX_KEY],
            payload,
        )
        return messages
