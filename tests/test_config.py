import os
import unittest

from omegaconf import OmegaConf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from numalogic._constants import TESTS_DIR
from numalogic.config import (
    ModelFactory,
    PreprocessFactory,
    PostprocessFactory,
    ThresholdFactory,
    NumalogicConf,
)
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.postprocess import TanhNorm
from numalogic.preprocess import LogTransformer

os.environ["OC_CAUSE"] = "1"


class TestNumalogicConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "resources", "config.yaml"))
        self.schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
        self.conf = OmegaConf.merge(self.schema, self._given_conf)

    def test_model(self):
        model_factory = ModelFactory()
        model = model_factory.get_model_instance(self.conf.model)
        self.assertIsInstance(model, SparseVanillaAE)
        self.assertEqual(model.seq_len, 20)

    def test_preprocess(self):
        preproc_factory = PreprocessFactory()
        preproc_clfs = []
        for _cfg in self.conf.preprocess:
            _clf = preproc_factory.get_model_instance(_cfg)
            preproc_clfs.append(_clf)

        preproc_pl = make_pipeline(*preproc_clfs)
        self.assertEqual(len(preproc_pl), 2)
        self.assertIsInstance(preproc_pl[0], LogTransformer)
        self.assertIsInstance(preproc_pl[1], StandardScaler)

    def test_threshold(self):
        thresh_factory = ThresholdFactory()
        thresh_clf = thresh_factory.get_model_instance(self.conf.threshold)
        self.assertIsInstance(thresh_clf, StdDevThreshold)

    def test_postprocess(self):
        postproc_factory = PostprocessFactory()
        postproc_clf = postproc_factory.get_model_instance(self.conf.postprocess)
        self.assertIsInstance(postproc_clf, TanhNorm)
        self.assertEqual(postproc_clf.scale_factor, 5)

    def test_trainer(self):
        trainer_cfg = self.conf.trainer
        trainer = AutoencoderTrainer(**trainer_cfg)
        self.assertIsInstance(trainer, AutoencoderTrainer)
        self.assertEqual(trainer.max_epochs, 40)


if __name__ == "__main__":
    unittest.main()
