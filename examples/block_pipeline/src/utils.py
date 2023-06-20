import logging
import os

from redis.sentinel import Sentinel

from numalogic.tools.types import Singleton, redis_client_t

_LOGGER = logging.getLogger(__name__)
_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.split(_DIR)[0]
TRAIN_DATA_PATH = os.path.join(_ROOT_DIR, "resources/train_data.csv")

AUTH = os.getenv("REDIS_AUTH")
HOST = os.getenv("REDIS_HOST", default="isbsvc-redis-isbs-redis-svc.default.svc")
PORT = os.getenv("REDIS_PORT", default="26379")
MASTERNAME = os.getenv("REDIS_MASTER_NAME", default="mymaster")


class RedisClient(metaclass=Singleton):
    """Singleton class to manage redis client."""

    _client: redis_client_t = None

    def __init__(self):
        if not self._client:
            self.set_client()

    def set_client(self) -> None:
        sentinel_args = {
            "sentinels": [(HOST, PORT)],
            "socket_timeout": 0.1,
        }
        _LOGGER.info("Connecting to redis sentinel: %s, %s, %s", sentinel_args, MASTERNAME, AUTH)
        sentinel = Sentinel(
            **sentinel_args,
            sentinel_kwargs=dict(password=AUTH),
            password=AUTH,
        )
        self._client = sentinel.master_for(MASTERNAME)

    def get_client(self) -> redis_client_t:
        return self._client
