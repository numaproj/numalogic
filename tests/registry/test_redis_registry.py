import unittest
from contextlib import contextmanager

import fakeredis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import RedisRegistry
from numalogic.tools.exceptions import ModelKeyNotFound


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

    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException("{} raised".format(exc_type.__name__))

    def test_construct_key(self):
        key = RedisRegistry.construct_key(["model_", "nnet"], ["error1"], ["prod"])
        self.assertEqual("model_:nnet::error1::prod", key)

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
        self.assertEqual(data.extras["model_version"], version)

    def test_load_model_with_metadata(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model, **{"lr": 0.01}
        )
        data = self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertIsNotNone(data.artifact)
        self.assertIsNotNone(data.metadata)
        self.assertEqual(data.extras["model_version"], version)

    def test_load_model_with_version(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        data = self.registry.load(
            skeys=self.skeys, dkeys=self.dkeys, version=version, production=False
        )
        self.assertIsNotNone(data.artifact)
        self.assertIsNone(data.metadata)
        self.assertEqual(data.extras["model_version"], version)

    def test_both_version_latest_model_with_version(self):
        with self.assertLogs(level="ERROR") as log:
            self.registry.load(skeys=self.skeys, dkeys=self.dkeys, production=False)
            self.assertTrue(log.output)

    def test_load_model_when_no_model(self):
        self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertRaises(ModelKeyNotFound)

    def test_load_model_when_model_stale(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        self.registry.delete(skeys=self.skeys, dkeys=self.dkeys, version=str(version))
        self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertRaises(ModelKeyNotFound)

    def test_delete_version(self):
        version = self.registry.save(
            skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model
        )
        self.registry.delete(skeys=self.skeys, dkeys=self.dkeys, version=str(version))
        self.registry.load(skeys=self.skeys, dkeys=self.dkeys)
        self.assertRaises(ModelKeyNotFound)

    def test_delete_model_not_in_registry(self):
        self.registry.save(skeys=self.skeys, dkeys=self.dkeys, artifact=self.pytorch_model)
        self.registry.delete(skeys=self.skeys, dkeys=self.dkeys, version=str(8))
        self.assertRaises(ModelKeyNotFound)