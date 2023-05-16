import os
import time
import uuid
from typing import Optional

import numpy as np
import numpy.typing as npt
from orjson import orjson
from pynumaflow.function import Datum
from redis.exceptions import RedisError, RedisClusterException

from anomalydetection import get_logger, DataStreamConf
from anomalydetection.entities import StreamPayload, Status, Header, WindowPayload
from anomalydetection.clients.sentinel import get_redis_client
from anomalydetection.tools import msg_forward
from anomalydetection.watcher import ConfigManager

_LOGGER = get_logger(__name__)

AUTH = os.getenv("REDIS_AUTH")


# TODO get the replacement value from config
def _clean_arr(
        id_: str,
        unique_key: str,
        arr: npt.NDArray[float],
        replace_val: float = 0.0,
        inf_replace: float = 1e10,
) -> npt.NDArray[float]:
    if not np.isfinite(arr).any():
        _LOGGER.warning(
            "%s - Non finite values encountered: %s for keys: %s", id_, list(arr), unique_key
        )
    return np.nan_to_num(arr, nan=replace_val, posinf=inf_replace, neginf=-inf_replace)


def __aggregate_window(
        key: str, ts: str,
        values: [tuple[str, float]],
        win_size: int,
        buff_size: int,
        recreate: bool
) -> list[tuple[[float], float]]:
    """
    Adds an element to the sliding window using a redis sorted set.

    Returns an empty list if adding the element does not create a new entry
    to the set.
    """
    redis_conf = ConfigManager.get_redis_config()
    redis_client = get_redis_client(
        redis_conf.host,
        redis_conf.port,
        password=AUTH,
        mastername=redis_conf.master_name,
        recreate=recreate,
    )

    value = "::".join([f"{key}:{val}" for key, val in values])
    with redis_client.pipeline() as pl:
        pl.zadd(key, {f"{value}::{ts}": ts})
        pl.zremrangebyrank(key, -(buff_size + 10), -buff_size)
        pl.zrange(key, -win_size, -1, withscores=True, score_cast_func=int)
        out = pl.execute()
    _is_new, _, _window = out
    if not _is_new:
        return []
    print(_window)
    _window = list(map(lambda x: (x[0].split("::")[:-1], x[1]), _window))
    return _window


def _get_windowed_data(msg, ds_conf: DataStreamConf) -> Optional[WindowPayload]:
    win_size = ds_conf.window_size
    buff_size = int(os.getenv("BUFF_SIZE", 10 * win_size))

    if buff_size < win_size:
        raise ValueError(
            f"Redis list buffer size: {buff_size} is less than window length: {win_size}"
        )

    values = [(key, float(val)) for key, val in msg.items() if key != "timestamp" and key != "unique_key"]
    unique_key = msg["unique_key"]

    # Create sliding window
    try:
        elements = __aggregate_window(
            unique_key, msg["timestamp"], values, win_size, buff_size, recreate=False
        )
    except (RedisError, RedisClusterException) as warn:
        _LOGGER.warning("Redis connection failed, recreating the redis client, err: %r", warn)
        elements = __aggregate_window(
            unique_key, msg["timestamp"], values, win_size, buff_size, recreate=True
        )

    # Drop message if no of elements is less than sequence length needed
    if len(elements) < win_size:
        return None

    # Construct payload object
    _uuid = uuid.uuid4().hex
    win_list = [float(_val) for _val, _ in elements]
    win_list = [float(val) for (_, val), _ in elements]
    metrics_list = [key for (key, _), _ in elements]

    # Store win_arr as a matrix with columns representing features
    win_arr = np.asarray(win_list).reshape(-1, 1)
    win_arr = _clean_arr(_uuid, unique_key, win_arr)

    return WindowPayload(timestamps=[str(_ts) for _, _ts in elements], data=win_arr, raw_data=win_arr.copy(), metrics=metrics_list)


@msg_forward
def window(_: str, datum: Datum) -> Optional[bytes]:
    """
    UDF to construct windowing of the streaming input data, required by ML models.
    """
    _start_time = time.perf_counter()

    _LOGGER.debug("Received Msg: %s ", datum.value)
    msg = orjson.loads(datum.value)

    unique_key = msg["unique_key"]
    config_name = unique_key.split(":")[0]
    ds_conf = ConfigManager.get_datastream_config(config_name)

    # Create sliding window
    sliding_window = _get_windowed_data(msg, ds_conf)

    if not sliding_window:
        return None

    payload = StreamPayload(
        uuid=uuid.uuid4().hex,
        unique_key=unique_key,
        window=sliding_window,
        header=Header.MODEL_INFERENCE,
        status=Status.EXTRACTED,
    )

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in window: %.4f sec", payload.uuid, time.perf_counter() - _start_time
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
