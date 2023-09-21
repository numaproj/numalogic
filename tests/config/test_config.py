# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
    ModelInfo,
)
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.models.autoencoder.variants import SparseVanillaAE, SparseConv1dAE, LSTMAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.transforms import LogTransformer, TanhNorm
from numalogic.tools.exceptions import UnknownConfigArgsError

os.environ["OC_CAUSE"] = "1"


class TestNumalogicConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "resources", "config.yaml"))
        self.schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
        self.conf = OmegaConf.merge(self.schema, self._given_conf)

    def test_model(self):
        model_factory = ModelFactory()
        model = model_factory.get_instance(self.conf.model)
        self.assertIsInstance(model, SparseVanillaAE)
        self.assertEqual(model.seq_len, 20)

    def test_preprocess(self):
        preproc_factory = PreprocessFactory()
        preproc_clfs = []
        for _cfg in self.conf.preprocess:
            _clf = preproc_factory.get_instance(_cfg)
            preproc_clfs.append(_clf)

        preproc_pl = make_pipeline(*preproc_clfs)
        self.assertEqual(len(preproc_pl), 2)
        self.assertIsInstance(preproc_pl[0], LogTransformer)
        self.assertIsInstance(preproc_pl[1], StandardScaler)

    def test_threshold(self):
        thresh_factory = ThresholdFactory()
        thresh_clf = thresh_factory.get_instance(self.conf.threshold)
        self.assertIsInstance(thresh_clf, StdDevThreshold)

    def test_postprocess(self):
        postproc_factory = PostprocessFactory()
        postproc_clf = postproc_factory.get_instance(self.conf.postprocess)
        self.assertIsInstance(postproc_clf, TanhNorm)
        self.assertEqual(postproc_clf.scale_factor, 5)

    def test_trainer(self):
        trainer_cfg = self.conf.trainer
        trainer = AutoencoderTrainer(**trainer_cfg.pltrainer_conf)
        self.assertIsInstance(trainer, AutoencoderTrainer)
        self.assertEqual(trainer.max_epochs, 40)


class TestFactory(unittest.TestCase):
    def test_instance(self):
        factory = ModelFactory()
        model = factory.get_instance(
            ModelInfo(
                name="SparseConv1dAE",
                conf={"seq_len": 12, "in_channels": 2, "enc_channels": [8, 16]},
            )
        )
        self.assertIsInstance(model, SparseConv1dAE)

    def test_cls(self):
        factory = ModelFactory()
        model_cls = factory.get_cls("LSTMAE")
        print(model_cls)
        self.assertEqual(model_cls.__class__, LSTMAE.__class__)

    def test_instance_err(self):
        factory = ModelFactory()
        with self.assertRaises(UnknownConfigArgsError):
            factory.get_instance(ModelInfo(name="Random", conf={"seq_len": 12, "in_channels": 2}))

    def test_cls_err(self):
        factory = ModelFactory()
        with self.assertRaises(UnknownConfigArgsError):
            factory.get_cls("Random")


if __name__ == "__main__":
    unittest.main()
