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


import numpy as np
import torch
from numalogic.tools.exceptions import DataModuleError, InvalidDataShapeError
from numpy.typing import NDArray
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl


class StreamingDataset(IterableDataset):
    def __init__(self, data: NDArray[float], seq_len: int):
        if seq_len > len(data):
            raise ValueError(f"Sequence length: {seq_len} is more than data size: {len(data)}")

        if data.ndim != 2:
            raise InvalidDataShapeError(
                f"Input data should have dim=2, received shape: {data.shape}"
            )

        self._seq_len = seq_len
        self._data = data.astype(np.float32)

    def create_seq(self, x):
        idx = 0
        while idx < len(self._data) - self._seq_len + 1:
            yield x[idx : idx + self._seq_len]
            idx += 1

    def __iter__(self):
        # TODO implement multi worker iter
        return iter(self.create_seq(self._data))

    def __len__(self):
        return len(self._data) - self._seq_len + 1

    def __getitem__(self, idx: int):
        if idx >= len(self._data) - self._seq_len + 1:
            raise IndexError(f"{idx} out of bound!")
        return self._data[idx : idx + self._seq_len]


class TimeseriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_len: int,
        train_data: NDArray,
        val_data: NDArray = None,
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
        if stage == "fit":
            self.train_dataset = StreamingDataset(self.train_data, self.seq_len)
            if self.val_data is None:
                return
            self.val_dataset = StreamingDataset(self.val_data, self.seq_len)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_data is None:
            raise DataModuleError("Validation data is not provided!")
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    @staticmethod
    def unbatch_sequences(batched: Tensor) -> Tensor:
        output = batched[:, 0, :]
        return torch.vstack((output, batched[-1, 1::]))
