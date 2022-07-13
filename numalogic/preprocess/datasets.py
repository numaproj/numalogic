from typing import Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data: Union[pd.DataFrame, NDArray], seq_len: int, permute=True):
        self._seq_len = seq_len
        self.permute = permute
        data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
        self._seq_x = self.create_sequences(data)

    @property
    def data(self) -> Tensor:
        return self._seq_x

    def create_sequences(self, X_in: NDArray) -> Tensor:
        output = []
        if len(X_in) < self._seq_len:
            raise ValueError(f"Length of X_in: {len(X_in)} smaller than seq_len: {self._seq_len}")
        for idx in range(len(X_in) - self._seq_len + 1):
            output.append(X_in[idx : (idx + self._seq_len)])
        output = torch.tensor(np.stack(output), dtype=torch.float)
        if self.permute:
            return torch.permute(output, (0, 2, 1))
        return output

    def recover_shape(self, seq_x: NDArray) -> NDArray:
        if self.permute:
            output = seq_x[:, :, 0]
            return np.vstack((output, seq_x[-1, :, 1:].T))

        output = seq_x[:, 0, :]
        return np.vstack((output, seq_x[-1, 1::]))

    def __len__(self):
        return self._seq_x.shape[0]

    def __getitem__(self, idx: int):
        return self._seq_x[idx]
