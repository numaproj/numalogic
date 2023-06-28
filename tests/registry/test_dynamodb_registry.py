import time
import unittest
from moto import mock_dynamodb, mock_sts

from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import LocalLRUCache, ArtifactData
from numalogic.registry import DynamoDBRegistry

ROLE = "qwertyuiopasdfghjklzxcvbnm"
ENV = "dev"


@mock_sts
@mock_dynamodb
class TestDynamoDBRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pytorch_model = VanillaAE(10)

    def setUp(self) -> None:
        self.cache = LocalLRUCache(cachesize=4, ttl=300)
        self.registry = DynamoDBRegistry("test_table", role=ROLE)
        self.registry_with_cache = DynamoDBRegistry(
            "test_table", role=ROLE, cache_registry=self.cache
        )
        self.registry.create_table()

    def test_no_local_cache(self):
        self.assertIsNone(
            self.registry._save_in_cache(
                "key", ArtifactData(artifact=self.pytorch_model, extras={}, metadata={})
            )
        )
        self.assertIsNone(self.registry._load_from_cache("key"))
        self.assertIsNone(self.registry._clear_cache("key"))

    def test_with_local_cache(self):
        self.registry_with_cache._save_in_cache(
            "key", ArtifactData(artifact=self.pytorch_model, extras={}, metadata={})
        )
        self.assertIsNotNone(self.registry_with_cache._load_from_cache("key"))
        self.assertIsNotNone(self.registry_with_cache._clear_cache("key"))

    def test_pack_artifact_data(self):
        dynamodb_item = self.registry._pack_artifact_data(
            artifact=self.pytorch_model,
            version="10",
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        self.assertIsNotNone(dynamodb_item["artifact"])
        self.assertEqual(dynamodb_item["version"], "10")
        self.assertIsNotNone(dynamodb_item["metadata"])

    def test_unpack_artifact_data(self):
        dynamodb_item = self.registry._pack_artifact_data(
            artifact=self.pytorch_model,
            version="10",
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        artifact_data = self.registry._unpack_artifact_data(dynamodb_item)
        self.assertEqual(artifact_data.extras.get("version"), "10")
        self.assertGreater(artifact_data.extras.get("timestamp"), 0)
        self.assertIsNotNone(artifact_data.artifact)
        self.assertEqual(artifact_data.metadata.get("model_type"), "pytorch")
        self.assertEqual(artifact_data.metadata.get("model_name"), "VanillaAE")

    def test_construct_key(self):
        skeys = ["model_type", "model_name"]

        key = self.registry.construct_key(skeys)
        self.assertEqual(key, "model_type:model_name")

    def test_construct_version_key(self):
        skeys = ["model_type", "model_name"]
        key = self.registry.construct_key(skeys)

        vkey = self.registry._construct_version_key(key, 10)
        self.assertEqual(vkey, "v10__model_type:model_name")

        vkey = self.registry._construct_version_key(key)
        self.assertEqual(vkey, "v0__model_type:model_name")

    def test_is_artifact_stale(self):
        artifact_data = ArtifactData(
            artifact=self.pytorch_model,
            extras={"timestamp": time.time()},
            metadata={},
        )

        self.assertTrue(self.registry.is_artifact_stale(artifact_data, 0))
        self.assertFalse(self.registry.is_artifact_stale(artifact_data, 5))

    def test_save_single_artifact(self):
        skeys = ["model_type", "model_name"]
        dkeys = ["100", "abs"]

        # Insert the first verison
        version = self.registry.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=self.pytorch_model,
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        self.assertEqual(version, "1")

        artifact_data = self.registry.load(skeys, dkeys, latest=True)
        self.assertIsNotNone(artifact_data)
        self.assertEqual(artifact_data.extras.get("version"), "1")

        # There must be version 1
        artifact_data = self.registry.load(skeys, dkeys, latest=False, version="1")
        self.assertIsNotNone(artifact_data)

        # There must be no version 2
        artifact_data = self.registry.load(skeys, dkeys, latest=False, version="2")
        self.assertIsNone(artifact_data)

        # Clean up the inserted model
        self.registry.delete(skeys, dkeys, version="1")

    def test_save_multiple_artifacts(self):
        skeys = ["model_type", "model_name"]
        dkeys = ["100", "abs"]

        # Insert the first verison
        self.registry.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=self.pytorch_model,
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        # Insert a second version
        self.registry.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=self.pytorch_model,
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        # Insert a third version
        version = self.registry.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=self.pytorch_model,
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        self.assertEqual(version, "3")

        artifact_data = self.registry.load(skeys, dkeys, latest=True)
        self.assertIsNotNone(artifact_data)
        self.assertEqual(artifact_data.extras.get("version"), "3")

        # The oldest version must be gone
        artifact_data = self.registry.load(skeys, dkeys, latest=False, version="1")
        self.assertIsNone(artifact_data)

        artifact_data = self.registry.load(skeys, dkeys, latest=False, version="2")
        self.assertIsNotNone(artifact_data)

        artifact_data = self.registry.load(skeys, dkeys, latest=False, version="3")
        self.assertIsNotNone(artifact_data)

        # Clean up the inserted models
        self.registry.delete(skeys, dkeys, version="2")
        self.registry.delete(skeys, dkeys, version="3")

    def test_delete_single_artifact(self):
        skeys = ["model_type", "model_name"]
        dkeys = ["100", "abs"]

        # Insert the first verison
        self.registry.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=self.pytorch_model,
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        # Delete the first version
        self.registry.delete(skeys, dkeys, version="1")

        # There must be no version 1
        artifact_data = self.registry.load(skeys, dkeys, latest=False, version="1")
        self.assertIsNone(artifact_data)

        # Or latest version
        artifact_data = self.registry.load(skeys, dkeys, latest=True)
        self.assertIsNone(artifact_data)

    def test_delete_multiple_artifacts(self):
        skeys = ["model_type", "model_name"]
        dkeys = ["100", "abs"]

        # Insert the first verison
        self.registry.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=self.pytorch_model,
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        # Insert a second version
        self.registry.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=self.pytorch_model,
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        # Insert a third version
        self.registry.save(
            skeys=skeys,
            dkeys=dkeys,
            artifact=self.pytorch_model,
            metadata={
                "model_type": "pytorch",
                "model_name": "VanillaAE",
            },
        )

        # Delete the second version
        self.registry.delete(skeys, dkeys, version="3")

        # There must be no version 3
        artifact_data = self.registry.load(skeys, dkeys, latest=False, version="3")
        self.assertIsNone(artifact_data)

        # The latest version must be 2
        artifact_data = self.registry.load(skeys, dkeys, latest=True)
        self.assertIsNotNone(artifact_data)
        self.assertEqual(artifact_data.extras.get("version"), "2")

        # Delete version 2, there must be no more versions
        self.registry.delete(skeys, dkeys, version="2")

        artifact_data = self.registry.load(skeys, dkeys, latest=False, version="2")
        self.assertIsNone(artifact_data)
        artifact_data = self.registry.load(skeys, dkeys, latest=True)
        self.assertIsNone(artifact_data)

    def test_load_latest_version(self):
        skeys = ["model_type", "model_name"]
        dkeys = ["100", "abs"]

        try:
            self.registry.load(skeys, dkeys, latest=True, version="1")
            self.fail("Should have raised ValueError, you cannot have both latest=True and version")
        except ValueError:
            pass

        try:
            self.registry.load(skeys, dkeys, latest=False)
            self.fail(
                "Should have raised ValueError, you cannot have both latest=False and no version"
            )
        except ValueError:
            pass

        self.assertIsNone(self.registry.load(skeys, dkeys, latest=True))
        self.assertIsNone(self.registry.load(skeys, dkeys, latest=False, version="1"))
