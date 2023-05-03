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

import logging
from typing import Optional
from collections.abc import Generator, Iterator

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader

from numalogic.tools.exceptions import InvalidDataShapeError

_LOGGER = logging.getLogger(__name__)


class StreamingDataset(IterableDataset):
    r"""
    An iterable Dataset designed for streaming time series input.

    Args:
        data: A numpy array containing the input data in the shape of (batch, num_features).
        seq_len: Length of the sliding window sequences to be generated from the input data

    Raises:
        ValueError: If the sequence length is greater than the data size
        InvalidDataShapeError: If the input data array does not
                               have a minimum dimension size of 2
    """

    __slots__ = ("_seq_len", "_data")

    def __init__(self, data: npt.NDArray[float], seq_len: int):
        if seq_len > len(data):
            raise ValueError(f"Sequence length: {seq_len} is more than data size: {len(data)}")

        if data.ndim != 2:
            raise InvalidDataShapeError(
                f"Input data should have dim=2, received shape: {data.shape}"
            )

        self._seq_len = seq_len
        self._data = data.astype(np.float32)

    @property
    def data(self) -> npt.NDArray[float]:
        """
        Returns the reference data in the input shape
        """
        return self._data

    def create_seq(self, input_: npt.NDArray[float]) -> Generator[npt.NDArray[float], None, None]:
        r"""
        A generator function that yields sequences of specified length from the input data.
        Yields a subarray from the input data of size (seq_len, num_features)

        Args:
            input_: A numpy array containing the input data.
        """
        idx = 0
        while idx < len(self):
            yield input_[idx : idx + self._seq_len]
            idx += 1

    def __iter__(self) -> Iterator[npt.NDArray[float]]:
        r"""
        Returns an iterator for the StreamingDataset object.

        Raises:
            NotImplementedError: If multiple worker input is provided
        """
        worker_info = torch.utils.data.get_worker_info()
        if not worker_info or worker_info.num_workers == 1:
            return self.create_seq(self._data)

        raise NotImplementedError("Multiple workers are not supported yet for Streaming Dataset")

    def __len__(self) -> int:
        r"""
        Returns the number of sequences that can be generated from the input data.
        """
        return len(self._data) - self._seq_len + 1

    def __getitem__(self, idx: int) -> npt.NDArray[float]:
        r"""
        Retrieves a sequence from the input data at the specified index.
        """
        if idx >= len(self):
            raise IndexError(f"{idx} out of bound!")
        return self._data[idx : idx + self._seq_len]


class TimeseriesDataModule(pl.LightningDataModule):
    r"""
    A time series data module for use in PyTorch Lightning,
    using a StreamingDataset for training and validation datasets.

    Args:
        seq_len: The length of the sequences to be generated from the input data.
        data: A numpy array containing the training data in the shape of (batch, num_features)
        val_split_ratio: ratio of data to be used for validation set
        batch_size: The size of each batch of data. Defaults to 64.
    """

    def __init__(
        self,
        seq_len: int,
        data: npt.NDArray[float],
        val_split_ratio: float = 0.1,
        batch_size: int = 64,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data = data

        if val_split_ratio <= 0.0 or val_split_ratio >= 1.0:
            raise ValueError("val_split_ratio can only accept values between 0.0 and 1.0")

        self.val_split_ratio = val_split_ratio

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str) -> None:
        r"""
        Sets up the data module by initializing the train and validation datasets.
        """
        if stage == "fit":
            val_size = np.floor(self.val_split_ratio * len(self.data)).astype(int)
            _LOGGER.info("Size of validation set: %s", val_size)

            self.train_dataset = StreamingDataset(self.data[:-val_size, :], self.seq_len)
            self.val_dataset = StreamingDataset(self.data[-val_size:, :], self.seq_len)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        r"""
        Creates and returns a DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> Optional[EVAL_DATALOADERS]:
        r"""
        Creates and returns a DataLoader for the validation dataset if validation data is provided.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    @staticmethod
    def unbatch_sequences(batched: Tensor) -> Tensor:
        r"""
        Utility method to transform a 3D tensor of shape: (batch_size, seq_len, num_features)
        back into a shape of (new_batch, num_feautres).

        Note: This is an approximate inverse transormation as only the
        first element in seq_len is used for the first (new_batch - seq_len - 1) rows.
        """
        output = batched[:, 0, :]
        return torch.vstack((output, batched[-1, 1::]))
