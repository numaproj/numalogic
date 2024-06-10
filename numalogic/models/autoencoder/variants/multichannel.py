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


from collections.abc import Sequence

import torch
from torch import nn, Tensor

from numalogic.models.autoencoder.base import BaseAE

_empty_tensor = torch.empty(1)


class _ChannelAutoencoder(nn.Module):
    r"""Simple autoencoder running on a neural network channel in isolation from the other channels.

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features equals to the number of independent channels
        layersizes: encoder layer size
        dropout_p: the dropout value

    """

    def __init__(self, seq_len: int, n_features: int, layersizes: Sequence[int], dropout_p: float):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.dropout_p = dropout_p

        layers = self._construct_layers(layersizes)
        self.autoencoder = nn.Sequential(*layers)

    def _construct_layers(self, layersizes: Sequence[int]) -> nn.ModuleList:
        r"""Utility function to generate a simple feedforward network layer.

        Args:
        ----
            layersizes: layer size

        Returns
        -------
            A simple feedforward network layer of type nn.ModuleList
        """
        layers = nn.ModuleList()
        start_layersize = self.seq_len

        for lsize in layersizes:
            layers.extend(
                [
                    nn.Linear(start_layersize, lsize),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_p),
                ]
            )
            start_layersize = lsize

        layers.append(nn.Linear(layersizes[-1], self.seq_len))

        return layers

    def forward(self, x: Tensor) -> Tensor:
        return self.autoencoder(x)


class MultiChannelAE(BaseAE):
    r"""Channel Autoencoder model comprising Fully connected layers only.

    Args:
    ----
        signal_len: sequence length / window length
        n_channels: number of independent network channels
        n_features: num of features per channel
        encoder_layersizes: encoder layer size (default = Sequence[int] = (16, 8))
        decoder_layersizes: decoder layer size (default = Sequence[int] = (8, 16))
        dropout_p: the dropout value (default=0.25)
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_features: int = 1,
        layersizes: Sequence[int] = (16, 8, 16),
        dropout_p: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.dropout_prob = dropout_p
        self.n_features = n_features
        self.layersizes = layersizes

        # self.channels = []

        for i in range(self.n_channels):
            channel = _ChannelAutoencoder(
                seq_len=seq_len,
                n_features=n_features,
                layersizes=layersizes,
                dropout_p=dropout_p,
            )
            channel.apply(self.init_weights)
            setattr(self, f"channel{i}", channel)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize the parameters in the model."""
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        decoded_channels = []

        batch = torch.swapdims(batch, 1, 2)

        for i in range(self.n_channels):
            channel = getattr(self, f"channel{i}")

            batch_channel = batch[:, [i]]
            decoded = channel(batch_channel)
            decoded_channels.append(decoded)

        decoded_channels = torch.stack(decoded_channels, dim=-1)
        decoded_channels = torch.squeeze(decoded_channels, 1)

        return _empty_tensor, decoded_channels

    def _get_reconstruction_loss(self, batch: Tensor):
        _, recon = self.forward(batch)
        return self.criterion(batch, recon)

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        """Returns reconstruction for streaming input."""
        recon = self.reconstruction(batch)
        return self.criterion(batch, recon, reduction="none")
