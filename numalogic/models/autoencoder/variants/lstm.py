import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import calculate_gain

from numalogic.models.autoencoder.base import TorchAE
from numalogic.preprocess.datasets import SequenceDataset

_LOGGER = logging.getLogger(__name__)
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_LOGGER.info("Current device: %s", _DEVICE)


class _Encoder(nn.Module):
    r"""
    Encoder network for the autoencoder network.

    Args:
        seq_len: sequence length / window length,
        no_features: number of features
        embedding_size: embedding layer size
        num_layers: number of decoder layers
    """

    def __init__(self, seq_len: int, no_features: int, embedding_size: int, num_layers=1):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(
            input_size=no_features,
            hidden_size=embedding_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        _, (hidden_state, __) = self.lstm(x)
        return hidden_state[-1, :, :]


class _Decoder(nn.Module):
    r"""
    Decoder network for the autoencoder network.

    Args:
        seq_len: sequence length / window length,
        no_features: number of features
        hidden_size: hidden layer size(default = 32)
        num_layers: number of decoder layers
    """

    def __init__(
        self, seq_len: int, no_features: int, output_size: int, hidden_size=32, num_layers=1
    ):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (_, __) = self.lstm(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out


class LSTMAE(TorchAE):
    r"""
    Long Short-Term Memory (LSTM) based autoencoder.

    Args:
        seq_len: sequence length / window length,
        no_features: number of features
        embedding_dim: embedding dimension for the network
        encoder_layers: number of encoder layers (default = 1)
        decoder_layers: number of decoder layers (default = 1)

    """

    def __init__(
        self,
        seq_len: int,
        no_features: int,
        embedding_dim: int,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim

        self.encoder = _Encoder(
            seq_len=self.seq_len,
            no_features=self.no_features,
            embedding_size=self.embedding_dim,
            num_layers=encoder_layers,
        )
        self.encoder = self.encoder.to(_DEVICE)
        self.encoder.apply(self.init_weights)

        self.decoder = _Decoder(
            seq_len=self.seq_len,
            no_features=self.embedding_dim,
            output_size=self.no_features,
            hidden_size=embedding_dim,
            num_layers=decoder_layers,
        )
        self.decoder = self.decoder.to(_DEVICE)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        r"""
        Initiate parameters in the transformer model.
        """
        for node, param in m.named_parameters():
            if "bias" in node:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_normal_(param, gain=calculate_gain("tanh"))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def construct_dataset(self, x: Tensor, seq_len: int = None) -> SequenceDataset:
        r"""
         Constructs dataset given tensor and seq_len

         Args:
            x: Tensor type
            seq_len: sequence length / window length

        Returns:
            SequenceDataset type
        """
        __seq_len = seq_len or self.seq_len
        dataset = SequenceDataset(x, __seq_len, permute=False)
        return dataset
