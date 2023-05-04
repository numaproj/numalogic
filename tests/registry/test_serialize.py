import unittest

from sklearn.preprocessing import StandardScaler
from torchinfo import summary

from numalogic.registry._serialize import loads, dumps

from numalogic.models.autoencoder.variants import VanillaAE


class TestSerialize(unittest.TestCase):
    def test_dumps_loads1(self):
        model = VanillaAE(10)
        serialized_obj = dumps(model)
        deserialized_obj = loads(serialized_obj)
        self.assertEqual(str(summary(model)), str(summary(deserialized_obj)))

    def test_dumps_loads2(self):
        model = StandardScaler()
        model.mean_ = 1000
        serialized_obj = dumps(model)
        deserialized_obj = loads(serialized_obj)
        self.assertEqual(model.mean_, deserialized_obj.mean_)
