import os
import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from torch.testing import assert_close
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.tools.data import (
    StreamingDataset,
    TimeseriesDataModule,
    inverse_window,
    StreamingDataLoader,
)
from numalogic.tools.exceptions import InvalidDataShapeError

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
SEQ_LEN = 12
RNG = np.random.default_rng(42)


@pytest.fixture
def setup():
    m, n = 30, 3
    return np.arange(30 * 3).reshape(30, 3).astype(np.float32), m, n


class TestStreamingDataset:
    def test_dataset(self, setup):
        data, m, n = setup
        dataset = StreamingDataset(data, seq_len=SEQ_LEN)
        for seq in dataset:
            assert (SEQ_LEN, n) == seq.shape
        assert (data.shape[0] - SEQ_LEN + 1) == len(dataset)
        assert_allclose(np.ravel(dataset[0]), np.ravel(data[:12, :]))
        assert_allclose(data, dataset.data)

    def test_dataset_getitem(self, setup):
        data, m, n = setup
        ds = StreamingDataset(data, seq_len=SEQ_LEN)
        assert (len(data) - SEQ_LEN + 1) == len(ds)
        assert (15 - SEQ_LEN + 1, SEQ_LEN, n) == ds[:15].shape
        assert (1, SEQ_LEN, n) == ds[3:15].shape
        assert (m - SEQ_LEN + 1, SEQ_LEN, n) == ds[:50].shape

    def test_as_array(self, setup):
        data, m, n = setup
        ds = StreamingDataset(data, seq_len=SEQ_LEN)
        assert (m - SEQ_LEN + 1, SEQ_LEN, n) == ds.as_array().shape

    def test_w_dataloader_01(self, setup):
        data, m, n = setup
        batch_size = 4
        dl = DataLoader(
            StreamingDataset(data, seq_len=SEQ_LEN),
            batch_size=batch_size,
            num_workers=1,
            drop_last=True,
        )
        for idx, batch in enumerate(dl):
            assert_close(batch[0, 1, :], batch[1, 0, :])
            assert_close(batch[2, 1, :], batch[3, 0, :])

        with pytest.raises(NotImplementedError):
            dl = DataLoader(
                StreamingDataset(data, seq_len=SEQ_LEN),
                batch_size=batch_size,
                drop_last=True,
                num_workers=2,
            )
            for _ in dl:
                pass

    def test_dataset_err_01(self, setup):
        data, m, _ = setup
        with pytest.raises(ValueError):
            StreamingDataset(data, seq_len=m + 1)

    def test_dataset_err_02(self, setup):
        data, m, _ = setup
        dataset = StreamingDataset(data, seq_len=SEQ_LEN)
        with pytest.raises(IndexError):
            _ = dataset[m - 5]

    def test_dataset_err_03(self, setup):
        data, _, _ = setup
        with pytest.raises(InvalidDataShapeError):
            StreamingDataset(data.ravel(), seq_len=SEQ_LEN)

    def test_ds_with_stride(self, setup):
        data, m, n = setup
        stride = 4
        dataset = StreamingDataset(data, seq_len=SEQ_LEN, stride=stride)
        assert (m - SEQ_LEN) // stride + 1 == len(dataset)
        for idx, seq in enumerate(dataset):
            assert (SEQ_LEN, n) == seq.shape
            assert_array_equal(seq[0], data[idx * stride])

        assert (len(dataset), SEQ_LEN, n) == dataset.as_array().shape
        assert (len(dataset), SEQ_LEN, n) == dataset.as_tensor().shape

    def test_ds_with_stride_err(self, setup):
        data, _, _ = setup
        with pytest.raises(ValueError):
            StreamingDataset(data, seq_len=SEQ_LEN, stride=20)

    def test_ds_with_stride_dataloader(self, setup):
        data, m, n = setup
        stride = 2
        dataset = StreamingDataset(data, seq_len=SEQ_LEN, stride=stride)
        batch_size = 5
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
        )
        total_samples = 0
        for batch in dl:
            total_samples += batch.shape[0]
            assert (batch_size, SEQ_LEN, n) == batch.shape
        assert total_samples == len(dataset)


class TestStreamingDataLoader(unittest.TestCase):
    data = None
    m = None
    n = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.m = 30
        cls.n = 3
        cls.data = np.arange(cls.m * cls.n).reshape(30, 3)

    def test_dataloader(self):
        batch_size = 4
        dl = StreamingDataLoader(
            self.data,
            seq_len=SEQ_LEN,
            batch_size=batch_size,
            num_workers=1,
            drop_last=True,
        )
        for idx, batch in enumerate(dl):
            assert_close(batch[0, 1, :], batch[1, 0, :])
            assert_close(batch[2, 1, :], batch[3, 0, :])

        with self.assertRaises(TypeError):
            StreamingDataLoader(
                self.data,
                seq_len=SEQ_LEN,
                dataset=DataLoader(StreamingDataset(self.data, seq_len=SEQ_LEN)),
            )


class TestTimeSeriesDataModule(unittest.TestCase):
    train_data = None
    test_data = None
    m = None
    n = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.n = 3
        cls.train_data = RNG.random((100, cls.n))
        cls.test_data = RNG.random((20, cls.n))

    def test_datamodule(self):
        datamodule = TimeseriesDataModule(SEQ_LEN, self.train_data, val_split_ratio=0.2)
        datamodule.setup(stage="fit")
        self.assertIsInstance(datamodule.train_dataloader(), DataLoader)

        datamodule.setup(stage="validate")
        self.assertIsInstance(datamodule.val_dataloader(), DataLoader)

    def test_datamodule_err(self):
        with self.assertRaises(ValueError):
            TimeseriesDataModule(SEQ_LEN, self.train_data, val_split_ratio=1.2)


class TestInverseWindow(unittest.TestCase):
    train_data = None
    test_data = None
    m = None
    n = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.n = 3
        cls.train_data = RNG.random((100, cls.n))
        cls.test_data = RNG.random((20, cls.n))

    def test_inverse_window(self):
        ratio = 0.2
        datamodule = TimeseriesDataModule(
            SEQ_LEN, self.train_data, val_split_ratio=ratio, batch_size=256
        )
        datamodule.setup(stage="fit")

        val_size = int(ratio * len(self.train_data))

        for batch in datamodule.train_dataloader():
            unbatched = inverse_window(batch, method="keep_first")
            self.assertTupleEqual(self.train_data[:-val_size].shape, unbatched.shape)
            self.assertAlmostEqual(
                torch.mean(unbatched).item(), np.mean(self.train_data[:-val_size]), places=5
            )

        for batch in datamodule.val_dataloader():
            unbatched = inverse_window(batch, method="keep_last")
            self.assertTupleEqual(self.train_data[-val_size:].shape, unbatched.shape)
            self.assertAlmostEqual(
                torch.mean(unbatched).item(), np.mean(self.train_data[-val_size:]), places=5
            )

    def test_inverse_window_err(self):
        with self.assertRaises(ValueError):
            inverse_window(torch.tensor([1, 2, 3]), method="invalid_method")


if __name__ == "__main__":
    unittest.main()
