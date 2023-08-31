import logging
import os
from typing import Optional

from numalogic.connectors._config import RedisConf
from numalogic.tools.exceptions import EnvVarNotFoundError
from numalogic.tools.types import redis_client_t
from redis.backoff import ExponentialBackoff
from redis.exceptions import RedisClusterException, RedisError
from redis.retry import Retry
from redis.sentinel import Sentinel, MasterNotFoundError

_LOGGER = logging.getLogger(__name__)
SENTINEL_CLIENT: Optional[redis_client_t] = None


def get_redis_client(
    host: str,
    port: int,
    password: str,
    mastername: str,
    master_node: bool = True,
) -> redis_client_t:
    """
    Return a master redis client for sentinel connections, with retry.

    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        mastername: Redis sentinel master name
        master_node: Whether to use the master node or the slave nodes

    Returns
    -------
        Redis client instance
    """
    global SENTINEL_CLIENT

    if SENTINEL_CLIENT:
        return SENTINEL_CLIENT

    retry = Retry(
        ExponentialBackoff(),
        3,
        supported_errors=(
            ConnectionError,
            TimeoutError,
            RedisClusterException,
            RedisError,
            MasterNotFoundError,
        ),
    )

    conn_kwargs = {
        "socket_timeout": 1,
        "socket_connect_timeout": 1,
        "socket_keepalive": True,
        "health_check_interval": 10,
    }

    sentinel = Sentinel(
        [(host, port)],
        sentinel_kwargs=dict(password=password, **conn_kwargs),
        retry=retry,
        password=password,
        **conn_kwargs
    )
    if master_node:
        SENTINEL_CLIENT = sentinel.master_for(mastername)
    else:
        SENTINEL_CLIENT = sentinel.slave_for(mastername)
    _LOGGER.info("Sentinel redis params: %s, master_node: %s", conn_kwargs, master_node)
    return SENTINEL_CLIENT


def get_redis_client_from_conf(redis_conf: RedisConf, **kwargs) -> redis_client_t:
    """
    Return a master redis client from config for sentinel connections, with retry.

    Args:
        redis_conf: RedisConf object with host, port, master_name, etc.
        **kwargs: Additional arguments to pass to get_redis_client.

    Returns
    -------
        Redis client instance
    """
    auth = os.getenv("REDIS_AUTH")
    if not auth:
        raise EnvVarNotFoundError("REDIS_AUTH not set!")

    return get_redis_client(
        redis_conf.url,
        redis_conf.port,
        password=os.getenv("REDIS_AUTH"),
        mastername=redis_conf.master_name,
        **kwargs
    )
