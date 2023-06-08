import json
import time
import uuid

import numpy as np
import pandas as pd
from typing import List

from numalogic.registry import RedisRegistry
from numalogic.tools.exceptions import RedisRegistryError
from pynumaflow.function import Datum, Messages, Message

from src import get_logger
from src.connectors.sentinel import get_redis_client_from_conf
from src.entities import Status, StreamPayload, Header
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)


class Preprocess:

    def __init__(self):
        self.model_registry = RedisRegistry(client=get_redis_client_from_conf())

    @classmethod
    def get_df(cls, data_payload: dict, features: List[str]) -> (pd.DataFrame, List[int]):
        timestamps = range(int(data_payload["start_time"]), int(data_payload["end_time"]) - 1, 60 * 1000)
        given_timestamps = [int(data["timestamp"]) for data in data_payload["data"]]

        rows = []
        for data in data_payload["data"]:
            feature_values = [float(data.get(feature)) for feature in features]
            rows.append(pd.Series(feature_values + [int(data["timestamp"])]))

        for timestamp in timestamps:
            if timestamp not in given_timestamps:
                rows.append(pd.Series([0] * len(features) + [timestamp]))

        df = pd.concat(rows, axis=1).T
        df.columns = features + ["timestamp"]
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df.fillna(0)
        df = df.drop('timestamp', axis=1)
        return df, [*timestamps]

    def preprocess(self, keys: List[str], metric: str, payload: StreamPayload) -> (np.ndarray, Status):
        preprocess_cfgs = ConfigManager.get_preprocess_config(config_name=keys[0], metric_name=metric)

        # Load preproc artifact
        try:
            preproc_artifact = self.model_registry.load(
                skeys=keys + [metric],
                dkeys=[_conf.name for _conf in preprocess_cfgs],
            )
        except RedisRegistryError as err:
            _LOGGER.exception(
                "%s - Error while fetching preproc artifact, keys: %s, metric: %s, err: %r",
                payload.uuid,
                keys,
                metric,
                err,
            )
            return None, Status.RUNTIME_ERROR

        # Check if artifact is found
        if not preproc_artifact:
            _LOGGER.info(
                "%s - Preprocess artifact not found, forwarding for static thresholding. Keys: %s, metric: %s",
                payload.uuid,
                keys,
                metric
            )
            return None, Status.ARTIFACT_NOT_FOUND

        # Perform preprocessing
        x_raw = payload.get_metric_arr(metric).reshape(-1, 1)
        preproc_clf = preproc_artifact.artifact
        x_scaled = preproc_clf.transform(x_raw)
        return x_scaled.flatten(), Status.PRE_PROCESSED

    def run(self, keys: List[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()
        _ = datum.event_time
        _ = datum.watermark
        _uuid = uuid.uuid4().hex
        _LOGGER.info("%s- Received Msg: { Keys: %s, Value: %s }", _uuid, keys, datum.value)

        messages = Messages()
        try:
            data_payload = json.loads(datum.value)
        except Exception as e:
            _LOGGER.error("Error while reading input json %r", e)
            messages.append(Message.to_drop())
            return messages

        # Load config
        stream_conf = ConfigManager.get_datastream_config(config_name=keys[0])
        raw_df, timestamps = self.get_df(data_payload, stream_conf.metrics)

        # Prepare payload for forwarding
        payload = StreamPayload(
            uuid=_uuid,
            composite_keys=keys,
            data=np.asarray(raw_df.values.tolist()),
            raw_data=np.asarray(raw_df.values.tolist()),
            metrics=raw_df.columns.tolist(),
            timestamps=timestamps,
            metadata=dict(),
        )

        if not np.isfinite(raw_df.values).any():
            _LOGGER.warning(
                "%s - Non finite values encountered: %s for keys: %s", payload.uuid, list(raw_df.values), keys
            )

        # Perform preprocessing for each metric
        for metric in payload.metrics:
            x_scaled, status = self.preprocess(keys, metric, payload)
            payload.set_status(metric=metric, status=status)

            # If preprocess failed, forward for static thresholding
            if x_scaled is None:
                payload.set_header(metric=metric, header=Header.STATIC_INFERENCE)
            else:
                payload.set_header(metric=metric, header=Header.MODEL_INFERENCE)
                payload.set_metric_data(metric=metric, arr=x_scaled)

        messages.append(Message(keys=keys, value=payload.to_json()))
        _LOGGER.info("%s - Sending Msg: { Keys: %s, Payload: %s }", payload.uuid, keys, payload)
        return messages
