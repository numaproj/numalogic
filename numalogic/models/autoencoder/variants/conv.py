import logging
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.init import calculate_gain
from torchinfo import summary

from numalogic.models.autoencoder.base import TorchAE
from numalogic.preprocess.datasets import SequenceDataset

LOGGER = logging.getLogger(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOGGER.info("Current device: %s", DEVICE)


class Conv1dAE(TorchAE):
    """
    One dimensional Convolutional Autoencoder with multichannel support.
    """

    def __init__(self, in_channels: int, enc_channels: int):
        super(Conv1dAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, enc_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(enc_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                enc_channels, enc_channels, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.BatchNorm1d(enc_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(enc_channels, in_channels, kernel_size=7, padding=3),
            nn.Upsample(scale_factor=2),
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        self.thresholds = None

    def __repr__(self) -> str:
        return summary(self)

    def summary(self, input_shape: Tuple[int, ...]):
        print(summary(self, input_size=input_shape))

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) in (nn.ConvTranspose1d, nn.Conv1d):
            nn.init.xavier_normal_(m.weight, gain=calculate_gain("relu"))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def construct_dataset(self, x: Tensor, seq_len: int = None) -> SequenceDataset:
        __seq_len = seq_len or self.seq_len
        dataset = SequenceDataset(x, __seq_len, permute=True)
        return dataset
