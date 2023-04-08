import os
import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose
from torch.testing import assert_close
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.tools.data import StreamingDataset, TimeseriesDataModule
from numalogic.tools.exceptions import InvalidDataShapeError

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
SEQ_LEN = 12


class TestStreamingDataset(unittest.TestCase):
    data = None
    m = None
    n = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.m = 30
        cls.n = 3
        cls.data = np.arange(cls.m * cls.n).reshape(30, 3)
        # cls.data = np.random.randn(cls.m, cls.n)

    def test_dataset(self):
        dataset = StreamingDataset(self.data, seq_len=SEQ_LEN)
        for seq in dataset:
            self.assertTupleEqual((SEQ_LEN, self.n), seq.shape)
        self.assertEqual(self.data.shape[0] - SEQ_LEN + 1, len(dataset))
        assert_allclose(np.ravel(dataset[0]), np.ravel(self.data[:12, :]))

    def test_w_dataloader(self):
        batch_size = 4
        dl = DataLoader(
            StreamingDataset(self.data, seq_len=SEQ_LEN),
            batch_size=batch_size,
            num_workers=1,
            drop_last=True,
        )
        for idx, batch in enumerate(dl):
            assert_close(batch[0, 1, :], batch[1, 0, :])
            assert_close(batch[2, 1, :], batch[3, 0, :])

        with self.assertRaises(NotImplementedError):
            dl = DataLoader(
                StreamingDataset(self.data, seq_len=SEQ_LEN),
                batch_size=batch_size,
                drop_last=True,
                num_workers=2,
            )
            for _ in dl:
                pass

    def test_dataset_err_01(self):
        with self.assertRaises(ValueError):
            StreamingDataset(self.data, seq_len=self.m + 1)

    def test_dataset_err_02(self):
        dataset = StreamingDataset(self.data, seq_len=SEQ_LEN)
        with self.assertRaises(IndexError):
            _ = dataset[self.m - 5]

    def test_dataset_err_03(self):
        with self.assertRaises(InvalidDataShapeError):
            StreamingDataset(self.data.ravel(), seq_len=SEQ_LEN)


class TestTimeSeriesDataModule(unittest.TestCase):
    train_data = None
    test_data = None
    m = None
    n = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.n = 3
        cls.train_data = np.random.randn(100, cls.n)
        cls.test_data = np.random.randn(20, cls.n)

    def test_datamodule(self):
        datamodule = TimeseriesDataModule(SEQ_LEN, self.train_data, val_split_ratio=0.2)
        datamodule.setup(stage="fit")
        self.assertIsInstance(datamodule.train_dataloader(), DataLoader)

        datamodule.setup(stage="validate")
        self.assertIsInstance(datamodule.val_dataloader(), DataLoader)

    def test_unbatch_sequences(self):
        ratio = 0.2
        datamodule = TimeseriesDataModule(
            SEQ_LEN, self.train_data, val_split_ratio=ratio, batch_size=256
        )
        datamodule.setup(stage="fit")

        val_size = int(ratio * len(self.train_data))

        for batch in datamodule.train_dataloader():
            unbatched = datamodule.unbatch_sequences(batch)
            self.assertTupleEqual(self.train_data[:-val_size].shape, unbatched.shape)
            self.assertAlmostEqual(
                torch.mean(unbatched).item(), np.mean(self.train_data[:-val_size]), places=5
            )

        for batch in datamodule.val_dataloader():
            unbatched = datamodule.unbatch_sequences(batch)
            self.assertTupleEqual(self.train_data[-val_size:].shape, unbatched.shape)
            self.assertAlmostEqual(
                torch.mean(unbatched).item(), np.mean(self.train_data[-val_size:]), places=5
            )


if __name__ == "__main__":
    unittest.main()
