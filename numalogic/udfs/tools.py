import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from numalogic.registry import ArtifactManager
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.tools.types import artifact_t, KEYS
from numalogic.udfs._config import StreamConf
from numalogic.udfs.entities import StreamPayload

_LOGGER = logging.getLogger(__name__)


def get_df(data_payload: dict, stream_conf: StreamConf) -> tuple[DataFrame, list]:
    """
    Function is used to create a dataframe from the data payload.
    Args:
        data_payload: data payload
        stream_conf: stream configuration.

    Returns
    -------
        dataframe and timestamps
    """
    features = stream_conf.metrics
    win_size = stream_conf.window_size
    df = (
        pd.DataFrame(data_payload["data"], columns=["timestamp", *features])
        .astype(float)
        .fillna(0)
        .tail(win_size)
    )
    return df[features], df["timestamp"].values.tolist()


def make_stream_payload(
    data_payload: dict, raw_df: DataFrame, timestamps: list, keys: list[str]
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
        data=np.asarray(raw_df.values.tolist()),
        raw_data=np.asarray(raw_df.values.tolist()),
        metrics=raw_df.columns.tolist(),
        timestamps=timestamps,
        metadata=data_payload["metadata"],
    )


def _load_model(
    skeys: KEYS, dkeys: KEYS, payload: StreamPayload, model_registry: ArtifactManager
) -> artifact_t:
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

    """
    try:
        preproc_artifact = model_registry.load(skeys, dkeys)
    except RedisRegistryError as err:
        _LOGGER.exception(
            "%s - Error while fetching preproc artifact, Keys: %s, Metrics: %s, Error: %r",
            payload.uuid,
            skeys,
            payload.metrics,
            err,
        )
        return None

    except Exception as ex:
        _LOGGER.exception(
            "%s - Unhandled exception while fetching preproc artifact, Keys: %s, Metric: %s,"
            " Error: %r",
            payload.uuid,
            payload.composite_keys,
            payload.metrics,
            ex,
        )
        return None
    else:
        return preproc_artifact