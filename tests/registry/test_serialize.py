import pickle
import timeit
import unittest

from sklearn.preprocessing import StandardScaler
from torchinfo import summary

from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry._serialize import loads, dumps


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

    def test_benchmark_state_dict_vs_model(self):
        model = VanillaAE(10, 2)
        serialized_sd = dumps(model.state_dict())
        serialized_obj = dumps(model)
        elapsed_obj = timeit.timeit(lambda: loads(serialized_obj), number=100)
        elapsed_sd = timeit.timeit(lambda: loads(serialized_sd), number=100)
        self.assertLess(elapsed_sd, elapsed_obj)

    def test_benchmark_default_vs_highest_protocol(self):
        model = VanillaAE(10, 2)
        serialized_default = dumps(model, pickle_protocol=pickle.DEFAULT_PROTOCOL)
        serialized_highest = dumps(model, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        elapsed_default = timeit.timeit(lambda: loads(serialized_default), number=100)
        elapsed_highest = timeit.timeit(lambda: loads(serialized_highest), number=100)
        self.assertLess(elapsed_highest, elapsed_default)
