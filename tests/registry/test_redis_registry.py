import logging
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import fakeredis
from freezegun import freeze_time
from redis import ConnectionError, InvalidResponse, TimeoutError
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import RedisRegistry, LocalLRUCache, ArtifactData
from numalogic.tools.exceptions import ModelKeyNotFound, RedisRegistryError
from numalogic.tools.types import KeyedArtifact

logging.basicConfig(level=logging.DEBUG)


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
        self.cache = LocalLRUCache(cachesize=4, ttl=1)
        self.registry = RedisRegistry(
            client=self.redis_client,
            cache_registry=self.cache,
        )
        self.registry_no_cache = RedisRegistry(client=self.redis_client)

    def tearDown(self) -> None:
        self.registry.client.flushall()
        self.registry_no_cache.client.flushall()
        self.cache.clear()
        LocalLRUCache.clear_instances()

    def test_no_cache(self):
        self.assertIsNone(
            self.registry_no_cache._save_in_cache(
                "key", ArtifactData(artifact=self.pytorch_model, extras={}, metadata={})
            )
        )
        self.assertIsNone(self.registry_no_cache._load_from_cache("key"))
        self.assertIsNone(self.registry_no_cache._clear_cache("key"))

    def test_construct_key(self):
        key = RedisRegistry.construct_key(["model_", "nnet"], ["error1"])
        self.assertEqual("model_:nnet::error1", key)

    def test_save_model_without_metadata_cache_hit(self):
        save_version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertEqual(data.extras["version"], save_version)
        resave_version1 = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model, **{"lr": 0.01}
        )
        resave_data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        print(resave_data.extras)
        self.assertEqual(save_version, "0")
        self.assertEqual(resave_version1, "1")
        self.assertEqual(resave_data.extras["version"], "0")

    def test_save_load_without_cache(self):
        save_version = self.registry_no_cache.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        data = self.registry_no_cache.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertEqual(data.extras["version"], save_version)
        resave_version1 = self.registry_no_cache.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        resave_data = self.registry_no_cache.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertEqual(save_version, "0")
        self.assertEqual(resave_version1, "1")
        self.assertEqual(resave_data.extras["version"], "1")

    def test_save_model_without_metadata_cache_miss(self):
        save_version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertEqual(data.extras["version"], save_version)
        resave_version1 = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        self.cache.clear()
        resave_data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertEqual(save_version, "0")
        self.assertEqual(resave_version1, "1")
        self.assertEqual(resave_data.extras["version"], "1")

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

    def test_delete_model(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        self.registry.delete(skeys=self.skeys, dkeys=self.dkeys, version=version)
        with self.assertRaises(ModelKeyNotFound):
            self.registry.load(skeys=self.skeys, dkeys=self.dkeys)

    def test_load_model_with_version(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys, version=version, latest=False)
        self.assertIsNotNone(data.artifact)
        self.assertIsNone(data.metadata)
        self.assertEqual(data.extras["version"], version)

    def test_check_if_model_stale_true(self):
        delta = datetime.today() - timedelta(days=5)
        with freeze_time(delta):
            self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertTrue(self.registry.is_artifact_stale(data, 12))

    def test_check_if_model_stale_false(self):
        delta = datetime.today()
        with freeze_time(delta):
            self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)
        with freeze_time(delta + timedelta(hours=7)):
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

    def test_load_latest_model_twice(self):
        with freeze_time(datetime.today() - timedelta(days=5)):
            self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)

        artifact_data_1 = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        artifact_data_2 = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertTrue(self.registry.is_artifact_stale(artifact_data_1, 4))
        self.assertEqual("registry", artifact_data_1.extras["source"])
        self.assertEqual("cache", artifact_data_2.extras["source"])

    def test_load_latest_cache_ttl_expire(self):
        self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)
        artifact_data_1 = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        time.sleep(1)
        artifact_data_2 = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertEqual("registry", artifact_data_1.extras["source"])
        self.assertEqual("registry", artifact_data_2.extras["source"])

    def test_multiple_save(self):
        self.registry.save_multiple(
            skeys=self.skeys,
            dict_artifacts={
                "AE": KeyedArtifact(dkeys=["AE"], artifact=VanillaAE(10)),
                "scaler": KeyedArtifact(dkeys=["scaler"], artifact=StandardScaler()),
            },
            **{"a": "b"}
        )
        artifact_data = self.registry.load(skeys=self.skeys, dkeys=["AE"])
        self.assertEqual("registry", artifact_data.extras["source"])
        self.assertIsNotNone(artifact_data.artifact)

    def test_load_non_latest_model_twice(self):
        old_version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)

        artifact_data_1 = self.registry.load(
            skeys=self.skeys, dkeys=self.dkeys, latest=False, version=old_version
        )
        artifact_data_2 = self.registry.load(
            skeys=self.skeys, dkeys=self.dkeys, latest=False, version=old_version
        )
        self.assertEqual("registry", artifact_data_1.extras["source"])
        self.assertEqual("cache", artifact_data_2.extras["source"])

    def test_delete_version(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        with self.assertRaises(ModelKeyNotFound):
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

    @patch("redis.Redis.set", Mock(side_effect=ConnectionError))
    def test_exception_call4(self):
        with self.assertRaises(RedisRegistryError):
            self.registry.save_multiple(
                skeys=self.skeys,
                dict_artifacts={"AE": KeyedArtifact(dkeys=self.dkeys, artifact=VanillaAE(10))},
            )
