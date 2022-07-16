from abc import ABCMeta, abstractmethod
from typing import Tuple

from torch import nn, Tensor
from torch.utils.data import Dataset
from torchinfo import summary


class TorchAE(nn.Module, metaclass=ABCMeta):
    """
    Abstract Base class for all Pytorch based autoencoder models for time-series data.
    """

    def __repr__(self) -> str:
        return str(summary(self))

    def summary(self, input_shape: Tuple[int, ...]) -> None:
        print(summary(self, input_size=input_shape))

    @abstractmethod
    def construct_dataset(self, x: Tensor, seq_len: int = None) -> Dataset:
        """
        Returns a dataset instance to be used for training.
        Needs to be overridden.
        """
        pass
