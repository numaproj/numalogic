import json
import os
import time
from json import JSONDecodeError
from typing import List

import numpy as np
import pandas as pd
from numalogic.registry import RedisRegistry, LocalLRUCache
from numalogic.tools.exceptions import RedisRegistryError
from pynumaflow.function import Datum, Messages, Message

from src import get_logger
from src.connectors.sentinel import get_redis_client_from_conf
from src.entities import Status, StreamPayload, Header
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", 3600))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", 10000))


class Preprocess:
    def __init__(self):
        local_cache = LocalLRUCache(cachesize=LOCAL_CACHE_SIZE, ttl=LOCAL_CACHE_TTL)
        self.model_registry = RedisRegistry(
            client=get_redis_client_from_conf(master_node=False), cache_registry=local_cache
        )
        self.config_manager = ConfigManager()

    @classmethod
    def get_df(
        cls, data_payload: dict, features: List[str], win_size: int
    ) -> (pd.DataFrame, List[int]):
        df = (
            pd.DataFrame(data_payload["data"], columns=["timestamp", *features])
            .astype(float)
            .fillna(0)
        )
        df.index = df.timestamp.astype(int)
        timestamps = np.arange(
            int(data_payload["start_time"]), int(data_payload["end_time"]) + 6e4, 6e4, dtype="int"
        )[-win_size:]
        df = df.reindex(timestamps, fill_value=0)
        return df[features], timestamps

    def preprocess(self, keys: list[str], payload: StreamPayload) -> (np.ndarray, Status):
        preprocess_cfgs = self.config_manager.get_preprocess_config(config_id=keys[0])

        # Load preproc artifact
        try:
            preproc_artifact = self.model_registry.load(
                skeys=keys,
                dkeys=[_conf.name for _conf in preprocess_cfgs],
            )
        except RedisRegistryError as err:
            _LOGGER.error(
                "%s - Error while fetching preproc artifact, Keys: %s, Metrics: %s, Error: %r",
                payload.uuid,
                keys,
                payload.metrics,
                err,
            )
            return None, Status.RUNTIME_ERROR

        except Exception as ex:
            _LOGGER.exception(
                "%s - Unhandled exception while fetching preproc artifact, Keys: %s, Metric: %s, Error: %r",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
                ex,
            )
            return None, Status.RUNTIME_ERROR

        # Check if artifact is found
        if not preproc_artifact:
            _LOGGER.info(
                "%s - Preprocess artifact not found, forwarding for static thresholding. Keys: %s, Metrics: %s",
                payload.uuid,
                keys,
                payload.metrics,
            )
            return None, Status.ARTIFACT_NOT_FOUND

        # Perform preprocessing
        x_raw = payload.get_data()
        preproc_clf = preproc_artifact.artifact
        x_scaled = preproc_clf.transform(x_raw)
        _LOGGER.info(
            "%s - Successfully preprocessed, Keys: %s, Metrics: %s, x_scaled: %s",
            payload.uuid,
            keys,
            payload.metrics,
            list(x_scaled),
        )
        return x_scaled, Status.PRE_PROCESSED

    def run(self, keys: List[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()
        _LOGGER.debug("Received Msg: { Keys: %s, Value: %s }", keys, datum.value)

        try:
            data_payload = json.loads(datum.value)
        except JSONDecodeError as e:
            _LOGGER.error("%s - Error while reading input json %r", e)
            return Messages(Message.to_drop())

        # Load config
        stream_conf = self.config_manager.get_stream_config(config_id=data_payload["config_id"])
        raw_df, timestamps = self.get_df(data_payload, stream_conf.metrics, stream_conf.window_size)

        if raw_df.shape[0] < stream_conf.window_size or raw_df.shape[1] != len(stream_conf.metrics):
            _LOGGER.error(
                "Dataframe shape: (%f, %f) less than window_size %f ",
                raw_df.shape[0],
                raw_df.shape[1],
                stream_conf.window_size,
            )
            return Messages(Message.to_drop())

        # Prepare payload for forwarding
        payload = StreamPayload(
            uuid=data_payload["uuid"],
            config_id=data_payload["config_id"],
            composite_keys=keys,
            data=np.ascontiguousarray(raw_df.to_numpy(), dtype=np.float32),
            raw_data=np.ascontiguousarray(raw_df.to_numpy(), dtype=np.float32),
            metrics=raw_df.columns.tolist(),
            timestamps=timestamps,
            metadata=data_payload["metadata"],
        )

        # TODO more optimal way to check for non finite values
        if not np.isfinite(raw_df.to_numpy()).any():
            _LOGGER.warning(
                "%s - Non finite values encountered: %s for keys: %s",
                payload.uuid,
                list(raw_df.values),
                keys,
            )

        # Perform preprocessing
        x_scaled, status = self.preprocess(keys, payload)
        payload.set_status(status=status)

        # If preprocess failed, forward for static thresholding
        if x_scaled is None:
            payload.set_header(header=Header.STATIC_INFERENCE)
        else:
            payload.set_header(header=Header.MODEL_INFERENCE)
            payload.set_data(arr=x_scaled)

        _LOGGER.info(
            "%s - Sending Msg: { Keys: %s, Metrics: %r }",
            payload.uuid,
            payload.composite_keys,
            payload.metrics,
        )
        _LOGGER.debug(
            "%s - Time taken in preprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        return Messages(Message(keys=keys, value=payload.to_json()))
