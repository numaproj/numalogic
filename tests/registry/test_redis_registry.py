import unittest
from unittest.mock import Mock, patch

import fakeredis
from freezegun import freeze_time
from redis import ConnectionError, InvalidResponse, TimeoutError
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import RedisRegistry
from numalogic.tools.exceptions import ModelKeyNotFound, RedisRegistryError


class TestRedisRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pytorch_model = VanillaAE(10)
        cls.scaler = StandardScaler()
        cls.model_sklearn = RandomForestRegressor(n_estimators=5)
        cls.skeys = ["test"]
        cls.dkeys = ["error"]
        server = fakeredis.FakeServer()
        cls.redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)

    def setUp(self):
        self.registry = RedisRegistry(client=self.redis_client)

    def tearDown(self) -> None:
        self.registry.client.flushall()

    def test_construct_key(self):
        key = RedisRegistry.construct_key(["model_", "nnet"], ["error1"])
        self.assertEqual("model_:nnet::error1", key)

    def test_save_model_without_metadata(self):
        save_version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        resave_version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        self.assertEqual(save_version, "0")
        self.assertEqual(resave_version, "1")

    def test_load_model_without_metadata(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertIsNotNone(data.artifact)
        self.assertIsNone(data.metadata)
        self.assertEqual(data.extras["version"], version)

    def test_load_model_with_metadata(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model, **{"lr": 0.01}
        )
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertIsNotNone(data.artifact)
        self.assertIsNotNone(data.metadata)
        self.assertEqual(data.extras["version"], version)

    def test_load_model_with_version(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys, version=version, latest=False)
        self.assertIsNotNone(data.artifact)
        self.assertIsNone(data.metadata)
        self.assertEqual(data.extras["version"], version)

    def test_check_if_model_stale_true(self):
        with freeze_time("2023-05-08 12:30:00"):
            self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertTrue(self.registry.is_artifact_stale(data, 12))

    def test_check_if_model_stale_false(self):
        with freeze_time("2023-05-08 12:30:00"):
            self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)
        with freeze_time("2023-05-08 19:30:00"):
            data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
            self.assertFalse(self.registry.is_artifact_stale(data, 8))

    def test_check_if_model_stale_err(self):
        self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        data.extras = None
        with self.assertRaises(RedisRegistryError):
            self.registry.is_artifact_stale(data, 8)

    def test_both_version_latest_model_with_version(self):
        with self.assertRaises(ValueError):
            self.registry.load(skeys=self.skeys, dkeys=self.dkeys, latest=False)

    def test_load_model_with_wrong_version(self):
        with self.assertRaises(ModelKeyNotFound):
            self.registry.load(skeys=self.skeys, dkeys=self.dkeys, version=str(100), latest=False)

    def test_load_model_when_no_model(self):
        with self.assertRaises(ModelKeyNotFound):
            self.registry.load(skeys=self.skeys, dkeys=self.dkeys)

    def test_load_model_when_model_stale(self):
        with self.assertRaises(ModelKeyNotFound):
            version = self.registry.save(
                skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
            )
            self.registry.delete(skeys=self.skeys, dkeys=self.dkeys, version=str(version))
            self.registry.load(skeys=self.skeys, dkeys=self.dkeys)

    def test_delete_version(self):
        with self.assertRaises(ModelKeyNotFound):
            version = self.registry.save(
                skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
            )
            self.registry.delete(skeys=self.skeys, dkeys=self.dkeys, version=str(version))
            self.registry.load(skeys=self.skeys, dkeys=self.dkeys)

    def test_delete_model_not_in_registry(self):
        with self.assertRaises(ModelKeyNotFound):
            self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)
            self.registry.delete(skeys=self.skeys, dkeys=self.dkeys, version=str(8))

    @patch("redis.Redis.set", Mock(side_effect=ConnectionError))
    def test_exception_call1(self):
        with self.assertRaises(RedisRegistryError):
            self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)

    @patch("redis.Redis.get", Mock(side_effect=InvalidResponse))
    def test_exception_call2(self):
        with self.assertRaises(RedisRegistryError):
            self.registry.load(skeys=self.skeys, dkeys=self.dkeys)

    @patch("redis.Redis.exists", Mock(side_effect=TimeoutError))
    def test_exception_call3(self):
        with self.assertRaises(RedisRegistryError):
            self.registry.delete(skeys=self.skeys, dkeys=self.dkeys, version="0")
