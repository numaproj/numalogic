import json
from typing import Optional

from redis.cluster import RedisCluster
from redis.backoff import ExponentialBackoff
from redis.exceptions import RedisClusterException, RedisError
from redis.retry import Retry

from anomalydetection import get_logger
from anomalydetection.tools import is_host_reachable

_LOGGER = get_logger(__name__)
redis_client: Optional[RedisCluster] = None


def get_redis_client(
    host: str, port: str, password: str = None, decode_responses: bool = True, recreate=False
) -> RedisCluster:
    global redis_client

    if not recreate and redis_client:
        return redis_client

    redis_params = {
        "host": host,
        "port": port,
        "password": password,
        "decode_responses": decode_responses,
        "skip_full_coverage_check": True,
        "dynamic_startup_nodes": False,
        "cluster_error_retry_attempts": 3,
    }
    _LOGGER.info("Redis params: %s", json.dumps(redis_params, indent=4))

    if not is_host_reachable(host, port):
        _LOGGER.error("Redis Cluster is unreachable after retries!")

    retry = Retry(
        ExponentialBackoff(cap=2, base=1),
        3,
        supported_errors=(ConnectionError, TimeoutError, RedisClusterException, RedisError),
    )
    redis_client = RedisCluster(**redis_params, retry=retry)
    return redis_client
