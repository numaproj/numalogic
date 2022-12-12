import numpy as np
from numpy.typing import NDArray
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl


class StreamingDataset(IterableDataset):
    def __init__(self, data: NDArray, seq_len: int, permute=True):
        self._seq_len = seq_len
        self.permute = permute
        self._data = data.astype(np.float32)

    def create_seq(self, x):
        idx = 0
        while idx < len(self._data) - self._seq_len + 1:
            yield x[idx : idx + self._seq_len]
            idx += 1

    def __iter__(self):
        return iter(self.create_seq(self._data))

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx: int):
        if idx >= len(self._data) - self._seq_len + 1:
            raise IndexError(f"{idx} out of bound!")
        return self._data[idx : idx + self._seq_len]


class TimeseriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: NDArray[float],
        seq_len: int,
        val_data: NDArray[float] = None,
        batch_size: int = 64,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_data = train_data
        self.val_data = val_data

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str) -> None:
        self.train_dataset = StreamingDataset(self.train_data, self.seq_len)
        if self.val_data is not None:
            self.val_dataset = StreamingDataset(self.val_data, self.seq_len)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset:
            return DataLoader(self.val_dataset, batch_size=self.batch_size)
        return None

    @staticmethod
    def unbatch_sequences(batched):
        # output = batched[:, :, 0]
        # return np.vstack((output, batched[-1, :, 1:].T))

        output = batched[:, 0, :]
        return np.vstack((output, batched[-1, 1::]))
