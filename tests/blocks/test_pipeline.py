import os
import unittest

import pandas as pd
import torch
from fakeredis import FakeRedis, FakeServer
from sklearn.preprocessing import StandardScaler

from numalogic._constants import TESTS_DIR
from numalogic.blocks import BlockPipeline, PreprocessBlock, NNBlock, PostprocessBlock
from numalogic.blocks._transform import ThresholdBlock
from numalogic.models.autoencoder.variants import (
    VanillaAE,
    LSTMAE,
    Conv1dAE,
    TransformerAE,
    SparseVanillaAE,
    SparseConv1dAE,
)
from numalogic.models.threshold import StdDevThreshold
from numalogic.registry import RedisRegistry
from numalogic.transforms import TanhScaler, TanhNorm, LogTransformer

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
server = FakeServer()
SEQ_LEN = 10


class TestBlockPipeline(unittest.TestCase):
    x_train = None
    x_stream = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE, nrows=1000)
        df = df[["success", "failure"]]
        cls.x_train = df[:990].to_numpy()
        cls.x_stream = df[-10:].to_numpy()
        assert cls.x_train.shape == (990, 2)
        assert cls.x_stream.shape == (10, 2)

    def setUp(self) -> None:
        self.reg = RedisRegistry(client=FakeRedis(server=server))

    def test_pipeline_01(self):
        block_pl = BlockPipeline(
            PreprocessBlock(TanhScaler()),
            NNBlock(VanillaAE(SEQ_LEN, n_features=2), SEQ_LEN),
            ThresholdBlock(StdDevThreshold()),
            PostprocessBlock(TanhNorm()),
            registry=self.reg,
        )
        block_pl.fit(self.x_train, nn__max_epochs=1)
        out = block_pl(self.x_stream)

        self.assertTupleEqual(self.x_stream.shape, out.shape)
        self.assertEqual(4, len(block_pl))
        self.assertIsInstance(block_pl[1], NNBlock)

    def test_pipeline_02(self):
        block_pl = BlockPipeline(
            PreprocessBlock(StandardScaler()),
            NNBlock(LSTMAE(SEQ_LEN, no_features=2, embedding_dim=4), SEQ_LEN),
            PostprocessBlock(TanhNorm()),
            registry=self.reg,
        )
        block_pl.fit(
            self.x_train,
            nn__max_epochs=1,
            nn__accelerator="cpu",
        )
        out = block_pl.run(self.x_stream)
        self.assertTupleEqual(self.x_stream.shape, out.shape)

    def test_pipeline_03(self):
        block_pl = BlockPipeline(
            PreprocessBlock(LogTransformer(), stateful=False),
            NNBlock(Conv1dAE(SEQ_LEN, in_channels=2), SEQ_LEN),
            registry=self.reg,
        )
        block_pl.fit(
            self.x_train,
            nn__max_epochs=1,
            nn__accelerator="cpu",
        )
        out = block_pl.run(self.x_stream)
        self.assertTupleEqual(self.x_stream.shape, out.shape)

    def test_pipeline_04(self):
        block_pl = BlockPipeline(
            PreprocessBlock(StandardScaler()),
            NNBlock(TransformerAE(SEQ_LEN, n_features=2), SEQ_LEN),
            ThresholdBlock(StdDevThreshold()),
            registry=self.reg,
        )
        block_pl.fit(
            self.x_train,
            nn__max_epochs=1,
            nn__accelerator="cpu",
        )
        out = block_pl.run(self.x_stream)
        self.assertTupleEqual(self.x_stream.shape, out.shape)
        for block in block_pl:
            self.assertTrue(block.stateful)

    def test_pipeline_05(self):
        block_pl = BlockPipeline(
            PreprocessBlock(LogTransformer(), stateful=False),
            NNBlock(SparseVanillaAE(seq_len=SEQ_LEN, n_features=2), SEQ_LEN),
            PostprocessBlock(TanhNorm()),
            registry=self.reg,
        )
        block_pl.fit(
            self.x_train,
            nn__max_epochs=1,
        )
        out = block_pl.run(self.x_stream)
        self.assertTupleEqual(self.x_stream.shape, out.shape)

    def test_pipeline_persistence(self):
        skeys = ["test"]
        dkeys = ["pipeline"]
        # Pipeline for saving
        pl_1 = BlockPipeline(
            PreprocessBlock(TanhScaler()),
            NNBlock(SparseConv1dAE(seq_len=SEQ_LEN, in_channels=2), SEQ_LEN),
            ThresholdBlock(StdDevThreshold()),
            PostprocessBlock(TanhNorm()),
            registry=self.reg,
        )
        pl_1.fit(
            self.x_train,
            nn__accelerator="cpu",
            nn__max_epochs=1,
        )

        _preweights = []
        with torch.no_grad():
            for params in pl_1[1].artifact.parameters():
                _preweights.append(torch.mean(params))

        pl_1.save(skeys, dkeys)

        # Pipeline for loading
        pl_2 = BlockPipeline(
            PreprocessBlock(TanhScaler()),
            NNBlock(SparseConv1dAE(seq_len=SEQ_LEN, in_channels=2), SEQ_LEN),
            PostprocessBlock(TanhNorm()),
            registry=self.reg,
        )
        pl_2.load(skeys, dkeys)

        _postweights = []
        with torch.no_grad():
            for params in pl_2[1].artifact.parameters():
                _postweights.append(torch.mean(params))

        self.assertListEqual(_preweights, _postweights)
        out = pl_2(self.x_stream)
        self.assertTupleEqual(self.x_stream.shape, out.shape)

    def test_pipeline_save_err(self):
        block_pl = BlockPipeline(
            PreprocessBlock(TanhScaler()),
            NNBlock(VanillaAE(SEQ_LEN, n_features=2), SEQ_LEN),
            PostprocessBlock(TanhNorm()),
        )
        block_pl.fit(self.x_train, nn__max_epochs=1)
        self.assertRaises(ValueError, block_pl.save, ["ml"], ["pl"])

    def test_pipeline_fit_err(self):
        block_pl = BlockPipeline(
            PreprocessBlock(TanhScaler()),
            NNBlock(VanillaAE(SEQ_LEN, n_features=2), SEQ_LEN),
            PostprocessBlock(TanhNorm()),
        )
        self.assertRaises(ValueError, block_pl.fit, self.x_train, max_epochs=1)


if __name__ == "__main__":
    unittest.main()
