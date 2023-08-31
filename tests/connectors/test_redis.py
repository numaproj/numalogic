import unittest
from unittest.mock import patch

from redis.sentinel import Sentinel
import fakeredis

from numalogic.connectors.redis import get_redis_client

server = fakeredis.FakeServer()
fake_redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)


class TestRedisClient(unittest.TestCase):
    def test_sentinel_redis_client(self):
        with patch.object(Sentinel, "master_for", return_value=fake_redis_client):
            r = get_redis_client("hostname", 6379, "pass", "mymaster")
            self.assertTrue(r.ping())
