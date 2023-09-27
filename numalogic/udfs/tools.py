import logging
from dataclasses import replace
import time
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from redis import RedisError

from numalogic.registry import ArtifactManager, ArtifactData
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.tools.types import KEYS, redis_client_t
from numalogic.udfs._config import StreamConf
from numalogic.udfs.entities import StreamPayload

_LOGGER = logging.getLogger(__name__)


def get_df(
    data_payload: dict, stream_conf: StreamConf, fill_value: float = 0.0
) -> tuple[DataFrame, list[int]]:
    """
    Function is used to create a dataframe from the data payload.
    Args:
        data_payload: data payload
        stream_conf: stream configuration.
        fill_value: nan fill value.

    Returns
    -------
        dataframe and timestamps
    """
    features = stream_conf.metrics
    df = (
        pd.DataFrame(data_payload["data"], columns=["timestamp", *features])
        .fillna(fill_value)
        .tail(stream_conf.window_size)
    )
    return df[features].astype(np.float32), df["timestamp"].astype(int).tolist()


def make_stream_payload(
    data_payload: dict, raw_df: DataFrame, timestamps: list[int], keys: list[str]
) -> StreamPayload:
    """
    Make StreamPayload object
    Args:
        data_payload: payload dictionary
        raw_df: Dataframe
        timestamps: Timestamps
        keys: keys.

    Returns
    -------
        StreamPayload object
    """
    return StreamPayload(
        uuid=data_payload["uuid"],
        config_id=data_payload["config_id"],
        composite_keys=keys,
        data=np.ascontiguousarray(raw_df, dtype=np.float32),
        raw_data=np.ascontiguousarray(raw_df, dtype=np.float32),
        metrics=raw_df.columns.tolist(),
        timestamps=timestamps,
        metadata=data_payload["metadata"],
    )


# TODO: move to base NumalogicUDF class and look into payload mutation
def _load_artifact(
    skeys: KEYS,
    dkeys: KEYS,
    payload: StreamPayload,
    model_registry: ArtifactManager,
    load_latest: bool,
) -> tuple[Optional[ArtifactData], StreamPayload]:
    """
    Load artifact from redis
    Args:
        skeys: KEYS
        dkeys: KEYS
        payload: payload dictionary
        model_registry: registry where model is stored.

    Returns
    -------
    artifact_t object
    StreamPayload object

    """
    version_to_load = "-1"
    if payload.metadata and "artifact_versions" in payload.metadata:
        version_to_load = payload.metadata["artifact_versions"][":".join(dkeys)]
        _LOGGER.info("%s - Found version info for keys: %s, %s", payload.uuid, skeys, dkeys)
    else:
        _LOGGER.info(
            "%s - No version info passed on! Loading latest artifact version for Keys: %s",
            payload.uuid,
            skeys,
        )
        load_latest = True
    try:
        if load_latest:
            artifact = model_registry.load(skeys=skeys, dkeys=dkeys)
        else:
            artifact = model_registry.load(
                skeys=skeys, dkeys=dkeys, latest=False, version=version_to_load
            )
    except RedisRegistryError:
        _LOGGER.warning(
            "%s - Error while fetching artifact, Keys: %s, Metrics: %s",
            payload.uuid,
            skeys,
            payload.metrics,
        )
        return None, payload

    except Exception:
        _LOGGER.exception(
            "%s - Unhandled exception while fetching preproc artifact, Keys: %s, Metric: %s,",
            payload.uuid,
            payload.composite_keys,
            payload.metrics,
        )
        return None, payload
    else:
        _LOGGER.info(
            "%s - Loaded Model. Source: %s , version: %s, Keys: %s, %s",
            payload.uuid,
            artifact.extras.get("source"),
            artifact.extras.get("version"),
            skeys,
            dkeys,
        )
        if (
            artifact.metadata
            and "artifact_versions" in artifact.metadata
            and "artifact_versions" not in payload.metadata
        ):
            payload = replace(
                payload,
                metadata={
                    "artifact_versions": artifact.metadata["artifact_versions"],
                    **payload.metadata,
                },
            )
        return artifact, payload


class TrainMsgDeduplicator:
    """
    TrainMsgDeduplicator class is used to deduplicate the train messages.
    Args:
        r_client: Redis client.
    """

    __slots__ = "client"

    def __init__(self, r_client: redis_client_t):
        self.client = r_client

    @staticmethod
    def __construct_key(keys: KEYS) -> str:
        return f"TRAIN::{':'.join(keys)}"

    def __fetch_ts(self, key: str) -> tuple[Optional[str], Optional[str]]:
        try:
            data = self.client.hgetall(key)
        except RedisError:
            _LOGGER.exception("Problem  fetching ts information for the key: %s", key)
            return None, None
        else:
            # decode the key:value pair and update the values
            data = {key.decode(): data.get(key).decode() for key in data}
            _msg_read_ts = str(data["_msg_read_ts"]) if data and "_msg_read_ts" in data else None
            _msg_train_ts = str(data["_msg_train_ts"]) if data and "_msg_train_ts" in data else None
            return _msg_read_ts, _msg_train_ts

    def ack_read(self, key: KEYS, uuid: str, retrain_freq: int = 24, retry: int = 600) -> bool:
        """
        Acknowledge the read message. Return True wh`en the msg has to be trained.
        Args:
            key: key
            uuid: uuid.
            retrain_freq: retrain frequency for the model in hrs
            retry: Time difference(in secs) between triggering retraining and msg read_ack.

        Returns
        -------
            bool

        """
        _key = self.__construct_key(key)
        _msg_read_ts, _msg_train_ts = self.__fetch_ts(key=_key)
        if _msg_read_ts and time.time() - float(_msg_read_ts) < retry:
            _LOGGER.info("%s - Model with key : %s is being trained by another process", uuid, key)
            return False

        # This check is needed if there is backpressure in the pl.
        if _msg_train_ts and time.time() - float(_msg_train_ts) < retrain_freq * 60 * 60:
            _LOGGER.info(
                "%s - Model was saved for the key: %s in less than %s secs, skipping training",
                uuid,
                key,
                retrain_freq,
            )
            return False
        try:
            self.client.hset(name=_key, key="_msg_read_ts", value=str(time.time()))
        except RedisError:
            _LOGGER.exception(
                "%s - Problem while updating msg_read_ts information for the key: %s",
                uuid,
                key,
            )
            return False
        _LOGGER.info("%s - Acknowledging request for Training for key : %s", uuid, key)
        return True

    def ack_train(self, key: KEYS, uuid: str) -> bool:
        """
        Acknowledge the train message is trained and saved. Return True when
                _msg_train_ts is updated.
        Args:
            key: key
            uuid: uuid.

        Returns
        -------
            bool
        """
        _key = self.__construct_key(key)
        try:
            self.client.hset(name=_key, key="_msg_train_ts", value=str(time.time()))
        except RedisError:
            _LOGGER.exception(
                " %s - Problem while updating msg_train_ts information for the key: %s",
                uuid,
                key,
            )
            return False
        else:
            _LOGGER.info("%s - Acknowledging model saving complete for for the key: %s", uuid, key)
            return True
