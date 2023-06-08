import os
from typing import Optional

from numalogic.tools.types import redis_client_t
from redis.backoff import ExponentialBackoff
from redis.exceptions import RedisClusterException, RedisError
from redis.retry import Retry
from redis.sentinel import Sentinel, MasterNotFoundError

from src import get_logger
from src._config import RedisConf
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)
SENTINEL_MASTER_CLIENT: Optional[redis_client_t] = None


def get_redis_client(
    host: str,
    port: int,
    password: str,
    mastername: str,
    decode_responses: bool = False,
    recreate: bool = False,
) -> redis_client_t:
    """
    Return a master redis client for sentinel connections, with retry.

    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        mastername: Redis sentinel master name
        decode_responses: Whether to decode responses
        recreate: Whether to flush and recreate the client

    Returns:
        Redis client instance
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
    sentinel_args = {
        "sentinels": [(host, port)],
        "socket_timeout": 0.1,
        "decode_responses": decode_responses,
    }

    _LOGGER.info("Sentinel redis params: %s", sentinel_args)

    sentinel = Sentinel(
        **sentinel_args, sentinel_kwargs=dict(password=password), password=password, retry=retry
    )
    SENTINEL_MASTER_CLIENT = sentinel.master_for(mastername)
    return SENTINEL_MASTER_CLIENT


def get_redis_client_from_conf(redis_conf: RedisConf = None, **kwargs) -> redis_client_t:
    """
    Return a master redis client from config for sentinel connections, with retry.

    Args:
        redis_conf: RedisConf object with host, port, master_name, etc.
        **kwargs: Additional arguments to pass to get_redis_client.

    Returns:
        Redis client instance
    """
    if not redis_conf:
        redis_conf = ConfigManager.get_redis_config()

    return get_redis_client(
        redis_conf.host,
        redis_conf.port,
        password=os.getenv("REDIS_AUTH"),
        mastername=redis_conf.master_name,
        **kwargs
    )
