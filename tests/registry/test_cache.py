import time
import unittest

from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import LocalLRUCache, ArtifactData, ArtifactCache


class TestArtifactCache(unittest.TestCase):
    def test_cache(self):
        cache_reg = ArtifactCache(cachesize=2, ttl=2)
        with self.assertRaises(NotImplementedError):
            cache_reg.save("m1", ArtifactData(VanillaAE(10, 1), metadata={}, extras={}))
        with self.assertRaises(NotImplementedError):
            cache_reg.load("m1")
        with self.assertRaises(NotImplementedError):
            cache_reg.delete("m1")


class TestLocalLRUCache(unittest.TestCase):
    def test_cache_size(self):
        cache_registry = LocalLRUCache(cachesize=2, ttl=1)
        cache_registry.save("m1", ArtifactData(VanillaAE(10, 1), metadata={}, extras={}))
        cache_registry.save("m2", ArtifactData(VanillaAE(12, 1), metadata={}, extras={}))
        cache_registry.save("m3", ArtifactData(VanillaAE(14, 1), metadata={}, extras={}))

        self.assertIsNone(cache_registry.load("m1"))
        self.assertIsInstance(cache_registry.load("m2"), ArtifactData)
        self.assertIsInstance(cache_registry.load("m3"), ArtifactData)
        self.assertEqual(2, cache_registry.cachesize)
        self.assertEqual(1, cache_registry.ttl)

    def test_cache_overwrite(self):
        cache_registry = LocalLRUCache(cachesize=2, ttl=1)
        cache_registry.save(
            "m1", ArtifactData(VanillaAE(10, 1), metadata={}, extras=dict(version="1"))
        )
        cache_registry.save(
            "m1", ArtifactData(VanillaAE(12, 1), metadata={}, extras=dict(version="2"))
        )

        loaded_artifact = cache_registry.load("m1")
        self.assertDictEqual({"version": "2"}, loaded_artifact.extras)

    def test_cache_ttl(self):
        cache_registry = LocalLRUCache(cachesize=2, ttl=1)
        cache_registry.save("m1", ArtifactData(VanillaAE(10, 1), metadata={}, extras={}))
        self.assertIsInstance(cache_registry.load("m1"), ArtifactData)

        time.sleep(1)
        self.assertIsNone(cache_registry.load("m1"))

    def test_singleton(self):
        cache_registry_1 = LocalLRUCache(cachesize=2, ttl=1)
        cache_registry_1.save("m1", ArtifactData(VanillaAE(10, 1), metadata={}, extras={}))
        self.assertIsInstance(cache_registry_1.load("m1"), ArtifactData)

        cache_registry_2 = LocalLRUCache(cachesize=3, ttl=3)
        self.assertIsInstance(cache_registry_2.load("m1"), ArtifactData)

    def test_delete(self):
        cache_registry = LocalLRUCache(cachesize=2, ttl=1)
        cache_registry.save("m1", ArtifactData(VanillaAE(10, 1), metadata={}, extras={}))
        cache_registry.delete("m1")
        self.assertIsNone(cache_registry.load("m1"))


if __name__ == "__main__":
    unittest.main()
