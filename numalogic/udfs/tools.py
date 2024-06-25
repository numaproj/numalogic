import os
from dataclasses import replace
import time
from typing import Optional, NamedTuple
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame
from pynumaflow.mapper import Message


from numalogic.registry import ArtifactManager, ArtifactData
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.tools.types import KEYS, redis_client_t
from numalogic.udfs._config import StreamConf
from numalogic.udfs._logger import configure_logger
from numalogic.udfs._metrics_utility import _set_gauge, _increment_counter, _add_info
from numalogic.udfs.entities import StreamPayload, TrainerPayload

METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))

_struct_log = configure_logger()


class _DedupMetadata(NamedTuple):
    """Data Structure for Dedup Metadata."""

    msg_read_ts: Optional[str]
    msg_train_ts: Optional[str]
    msg_train_records: Optional[str]


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
    _conf = stream_conf.ml_pipelines[data_payload.get("pipeline_id", "default")]
    features = _conf.metrics
    df = (
        pd.DataFrame(data_payload["data"], columns=["timestamp", *features])
        .fillna(fill_value)
        .tail(stream_conf.window_size)
    )
    return df[features].astype(np.float32), df["timestamp"].astype(int).tolist()


def _update_gauge_metric(
    data: np.ndarray, metric_name: Sequence[str], labels: dict[str, str]
) -> None:
    """
    Utility function is used to update the gauge metric.
    Args:
        data: data
        metric_name: metric name in the payload
        labels: labels.
    """
    for _data, _metric_name in zip(data.T, metric_name):
        _set_gauge(
            gauge="RECORDED_DATA_GAUGE",
            labels=labels | {"metric_name": _metric_name},
            is_enabled=METRICS_ENABLED,
            data=np.mean(_data).squeeze(),
        )


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
        pipeline_id=data_payload.get("pipeline_id", "default"),
        composite_keys=keys,
        data=np.ascontiguousarray(raw_df, dtype=np.float32),
        raw_data=np.ascontiguousarray(raw_df, dtype=np.float32),
        metrics=raw_df.columns.tolist(),
        timestamps=timestamps,
        metadata=data_payload["metadata"],
    )


# TODO: move to base NumalogicUDF class and look into payload mutation
def _get_artifact_stats(artifact_data):
    return {
        "artifact_source": artifact_data.extras.get("source") or None,
        "version": artifact_data.extras.get("version") or None,
    }


def _load_artifact(
    skeys: KEYS,
    dkeys: KEYS,
    payload: StreamPayload,
    model_registry: ArtifactManager,
    load_latest: bool,
    vertex: str,
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
    _metric_label_values = {
        "source": payload.metadata["numalogic_opex_tags"]["source"],
        "vertex": vertex,
        "composite_key": ":".join(skeys),
        "config_id": payload.config_id,
        "pipeline_id": payload.pipeline_id,
    }

    logger = _struct_log.bind(
        uuid=payload.uuid,
        skeys=skeys,
        dkeys=dkeys,
        payload_metrics=payload.metrics,
        udf_vertex=vertex,
    )

    version_to_load = "-1"
    if payload.artifact_versions:
        artifact_version = payload.artifact_versions
        key = ":".join(dkeys)
        if key in artifact_version:
            version_to_load = artifact_version[key]
            logger.debug("Found version info for keys")
        else:
            logger.debug("Could not find what version of model to load")
    else:
        logger.debug(
            "No version info passed on! Loading latest artifact version, "
            "if one present in the registry"
        )
        load_latest = True
    try:
        if load_latest:
            artifact_data = model_registry.load(skeys=skeys, dkeys=dkeys)
        else:
            artifact_data = model_registry.load(
                skeys=skeys, dkeys=dkeys, latest=False, version=version_to_load
            )
    except RedisRegistryError:
        _increment_counter(
            "REDIS_ERROR_COUNTER", labels=_metric_label_values, is_enabled=METRICS_ENABLED
        )
        logger.warning("Error while fetching artifact")
        return None, payload

    except Exception:
        _increment_counter(
            "EXCEPTION_COUNTER", labels=_metric_label_values, is_enabled=METRICS_ENABLED
        )
        logger.exception("Unhandled exception while fetching artifact")
        return None, payload
    else:
        logger = logger.bind(
            artifact_source=artifact_data.extras.get("source"),
            artifact_version=artifact_data.extras.get("version"),
        )
        logger.debug("Loaded Model!")
        _increment_counter(
            counter="SOURCE_COUNTER",
            labels=({"artifact_source": artifact_data.extras.get("source")} | _metric_label_values),
            is_enabled=METRICS_ENABLED,
        )
        _add_info(
            info="MODEL_INFO",
            labels={
                "source": payload.metadata["numalogic_opex_tags"]["source"],
                "composite_key": ":".join(skeys),
                "config_id": payload.config_id,
                "pipeline_id": payload.pipeline_id,
            },
            data=_get_artifact_stats(artifact_data),
            is_enabled=METRICS_ENABLED,
        )
        if (
            artifact_data.metadata
            and "artifact_versions" in artifact_data.metadata
            and not payload.artifact_versions
        ):
            payload = replace(
                payload,
                artifact_versions=artifact_data.metadata["artifact_versions"],
            )
        return artifact_data, payload


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
    def __construct_train_key(keys: KEYS) -> str:
        _key = ":".join(keys)
        return f"TRAIN::{_key}"

    def __fetch_ts(self, uuid: str, key: str) -> _DedupMetadata:
        logger = _struct_log.bind(uuid=uuid, key=key)
        try:
            data = self.client.hgetall(key)
        except Exception:
            logger.exception("Problem  fetching ts information for the key")
            return _DedupMetadata(msg_read_ts=None, msg_train_ts=None, msg_train_records=None)
        else:
            # decode the key:value pair and update the values
            data = {key.decode(): data.get(key).decode() for key in data}
            _msg_read_ts = str(data["_msg_read_ts"]) if data and "_msg_read_ts" in data else None
            _msg_train_ts = str(data["_msg_train_ts"]) if data and "_msg_train_ts" in data else None
            _msg_train_records = (
                str(data["_msg_train_records"]) if data and "_msg_train_records" in data else None
            )
            return _DedupMetadata(
                msg_read_ts=_msg_read_ts,
                msg_train_ts=_msg_train_ts,
                msg_train_records=_msg_train_records,
            )

    def ack_insufficient_data(self, key: KEYS, uuid: str, train_records: int) -> bool:
        """
        Acknowledge the insufficient data message. Retry training after certain period of a time.
        Args:
            key: key
            uuid: uuid
            train_records: number of train records found.

        Returns
        -------
            bool.
        """
        _key = self.__construct_train_key(key)
        logger = _struct_log.bind(uuid=uuid, key=key)
        try:
            self.client.hset(name=_key, key="_msg_train_records", value=str(train_records))
        except Exception:
            logger.exception("Problem while updating _msg_train_records information for the key")
            return False
        logger.debug("Acknowledging insufficient data for the key")
        return True

    def ack_read(
        self,
        key: KEYS,
        uuid: str,
        retrain_freq: int = 24,
        retry: int = 600,
        min_train_records: int = 180,
        data_freq: int = 60,
    ) -> bool:
        """
        Acknowledge the read message. Return True when the msg has to be trained.
        Args:
            key: key
            uuid: uuid.
            retrain_freq: retrain frequency for the model in hrs
            retry: Time difference(in secs) between triggering retraining and msg read_ack.
            min_train_records: minimum number of records required for training.
            data_freq: data granularity/frequency in secs.

        Returns
        -------
            bool

        """
        _key = self.__construct_train_key(key)
        metadata = self.__fetch_ts(uuid=uuid, key=_key)
        logger = _struct_log.bind(uuid=uuid, key=key)
        _msg_read_ts, _msg_train_ts, _msg_train_records = (
            metadata.msg_read_ts,
            metadata.msg_train_ts,
            metadata.msg_train_records,
        )
        # If insufficient data: retry after (min_train_records-train_records) * data_granularity
        _curr_time = time.time()
        if (
            _msg_train_records
            and _msg_read_ts
            and _curr_time - float(_msg_read_ts)
            < (min_train_records - int(_msg_train_records)) * data_freq
        ):
            logger.debug(
                "There was insufficient data for the key in the past. Retrying fetching"
                " and training after secs",
                secs=((min_train_records - int(_msg_train_records)) * data_freq)
                - _curr_time
                + float(_msg_read_ts),
            )

            return False

        # Check if the model is being trained by another process
        if _msg_read_ts and time.time() - float(_msg_read_ts) < retry:
            logger.debug("Model with key is being trained by another process")
            return False

        # This check is needed if there is backpressure in the pipeline
        if _msg_train_ts and time.time() - float(_msg_train_ts) < retrain_freq * 60 * 60:
            logger.debug(
                "Model was saved for the key in less than retrain_freq hrs, skipping training",
                retrain_freq=retrain_freq,
            )
            return False
        try:
            self.client.hset(name=_key, key="_msg_read_ts", value=str(time.time()))
        except Exception:
            logger.exception("Problem while updating msg_read_ts information for the key")
            return False
        logger.debug("Acknowledging request for Training for key")
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
        _key = self.__construct_train_key(key)
        logger = _struct_log.bind(uuid=uuid, key=key)
        try:
            self.client.hset(name=_key, key="_msg_train_ts", value=str(time.time()))
        except Exception:
            logger.exception("Problem while updating msg_train_ts information for the key")
            return False
        logger.debug("Acknowledging model saving complete for the key")
        return True


def get_trainer_message(
    keys: list[str],
    stream_conf: StreamConf,
    payload: StreamPayload,
    **metric_values: dict,
) -> Message:
    """
    Get message for training request.

    Args:
    -------
        keys: List of keys
        stream_conf: StreamConf instance
        payload: StreamPayload object
        metric_values: Optional metric values for monitoring

    Returns
    -------
        Mapper Message instance
    """
    ckeys = [_ckey for _, _ckey in zip(stream_conf.composite_keys, payload.composite_keys)]
    train_payload = TrainerPayload(
        uuid=payload.uuid,
        composite_keys=ckeys,
        metrics=payload.metrics,
        config_id=payload.config_id,
        pipeline_id=payload.pipeline_id,
    )
    if metric_values:
        _increment_counter(
            counter="MODEL_STATUS_COUNTER",
            labels={"status": payload.status} | metric_values,
            is_enabled=METRICS_ENABLED,
        )
    _struct_log.debug(
        "Sending training request",
        uuid=payload.uuid,
        composite_keys=ckeys,
        metrics=payload.metrics,
        config_id=payload.config_id,
        pipeline_id=payload.pipeline_id,
    )
    return Message(keys=keys, value=train_payload.to_json(), tags=["train"])


def get_static_thresh_message(keys: list[str], payload: StreamPayload) -> Message:
    """
    Get message for static thresholding request.

    Args:
    -------
        keys: List of keys
        stream_conf: StreamConf instance
        payload: StreamPayload object

    Returns
    -------
        Mapper Message instance
    """
    _struct_log.debug(
        "Sending thresholding request",
        uuid=payload.uuid,
        keys=keys,
        metrics=payload.metrics,
        config_id=payload.config_id,
        pipeline_id=payload.pipeline_id,
    )
    return Message(keys=keys, value=payload.to_json(), tags=["staticthresh"])
