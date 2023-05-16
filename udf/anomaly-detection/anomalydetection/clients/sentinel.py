from typing import Optional

from redis.backoff import ExponentialBackoff
from redis.client import Redis
from redis.exceptions import RedisClusterException, RedisError
from redis.retry import Retry
from redis.sentinel import Sentinel, MasterNotFoundError

from anomalydetection import get_logger

_LOGGER = get_logger(__name__)
SENTINEL_MASTER_CLIENT: Optional[Redis] = None


def get_redis_client(
    host: str, port: int, password: str, mastername: str, recreate: bool = False
) -> Redis:
    """
    Return a master redis client for sentinel connections, with retry.
    """
    global SENTINEL_MASTER_CLIENT

    if not recreate and SENTINEL_MASTER_CLIENT:
        return SENTINEL_MASTER_CLIENT

    retry = Retry(
        ExponentialBackoff(cap=2, base=1),
        3,
        supported_errors=(
            ConnectionError,
            TimeoutError,
            RedisClusterException,
            RedisError,
            MasterNotFoundError,
        ),
    )
    sentinel_args = {"sentinels": [(host, port)], "socket_timeout": 0.1, "decode_responses": True}

    _LOGGER.info("Sentinel redis params: %s", sentinel_args)

    sentinel = Sentinel(
        **sentinel_args, sentinel_kwargs=dict(password=password), password=password, retry=retry
    )
    SENTINEL_MASTER_CLIENT = sentinel.master_for(mastername)
    return SENTINEL_MASTER_CLIENT
