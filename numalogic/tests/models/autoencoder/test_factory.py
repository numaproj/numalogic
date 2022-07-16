import unittest

from numalogic.models.autoencoder import ModelPlFactory
from numalogic.models.autoencoder.variants import VanillaAE


class TestModelPlFactory(unittest.TestCase):
    def test_get_pl_cls(self):
        self.assertEqual("AutoencoderPipeline", ModelPlFactory.get_pl_cls("ae").__name__)

    def test_get_pl_obj(self):
        model = VanillaAE(10, n_features=2)
        ae_pl = ModelPlFactory.get_pl_obj("ae_sparse", beta=0.1, model=model, seq_len=10)
        self.assertEqual("SparseAEPipeline", ae_pl.__class__.__name__)

    def test_get_pl_err(self):
        with self.assertRaises(NotImplementedError):
            ModelPlFactory.get_pl_obj("Whatever man!", seq_len=10)


if __name__ == "__main__":
    unittest.main()
