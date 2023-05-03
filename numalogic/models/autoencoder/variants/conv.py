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
from typing import Union
from collections.abc import Sequence

import torch
from torch import nn, Tensor
from torch.nn.init import calculate_gain

from numalogic.models.autoencoder.base import BaseAE

LOGGER = logging.getLogger(__name__)


def _get_activation_function(activation_name: str):
    if activation_name == "sigmoid":
        return nn.Sigmoid()
    if activation_name == "tanh":
        return nn.Tanh()
    if activation_name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation function provided: {activation_name}")


class ConvBlock(nn.Module):
    """
    Basic convolutional block consisting of:
    - convolutional layer
    - batch norm
    - relu activation
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.LazyConv1d(
            out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding
        )
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_: Tensor) -> Tensor:
        return self.relu(self.bnorm(self.conv(input_)))


class ConvTransposeBlock(nn.Module):
    """
    Basic transpose convolutional block consisting of:
    - transpose convolutional layer
    - batch norm
    - relu activation
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
    ):
        super().__init__()
        self.convtranspose = nn.LazyConvTranspose1d(
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
        )
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_: Tensor) -> Tensor:
        return self.relu(self.bnorm(self.convtranspose(input_)))


class Encoder(nn.Module):
    """
    Enoder network for Conv1dAE.
    """

    def __init__(
        self, num_channels: Sequence[int], kernel_sizes: Sequence[int], pool_kernel_size: int
    ):
        super().__init__()
        layers = self._construct_layers(num_channels, kernel_sizes, pool_kernel_size)
        self.encoder = nn.Sequential(*layers)

    @staticmethod
    def _construct_layers(
        num_filters: Sequence[int], kernel_sizes: Sequence[int], pool_k_size: int
    ):
        layers = nn.ModuleList()

        # Non-final layers
        for idx in range(len(num_filters) - 1):
            layers.extend(
                [
                    ConvBlock(
                        out_channels=num_filters[idx], kernel_size=kernel_sizes[idx], padding=1
                    ),
                    nn.MaxPool1d(pool_k_size),
                ]
            )
        # Latent layer
        layers.extend(
            [
                nn.LazyConv1d(
                    out_channels=num_filters[-1], kernel_size=kernel_sizes[-1], padding=1
                ),
                nn.ReLU(),
            ]
        )
        return layers

    def forward(self, input_: Tensor) -> Tensor:
        return self.encoder(input_)


class Decoder(nn.Module):
    """
    Decoder network for Conv1dAE.
    """

    def __init__(
        self,
        num_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        upsample_scale_factor: int,
        final_activation: str,
    ):
        super().__init__()
        layers = self._construct_layers(
            num_channels, kernel_sizes, final_activation, upsample_scale_factor
        )
        self.decoder = nn.Sequential(*layers)

    @staticmethod
    def _construct_layers(
        num_filters: Sequence[int],
        kernel_sizes: Sequence[int],
        final_activation: str,
        upscale_factor: int,
    ):
        layers = nn.ModuleList()

        # Non-final layers
        for idx in range(len(num_filters) - 1):
            layers.append(
                ConvTransposeBlock(
                    out_channels=num_filters[idx], kernel_size=kernel_sizes[idx], padding=1
                )
            )
            layers.append(nn.Upsample(scale_factor=upscale_factor, mode="linear"))

        # Output layer
        layers.append(
            nn.LazyConvTranspose1d(
                out_channels=num_filters[-1], kernel_size=kernel_sizes[-1], padding=1
            )
        )
        if final_activation:
            layers.append(_get_activation_function(final_activation))
        return layers

    def forward(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)


class Conv1dAE(BaseAE):
    r"""
    1D Convolutional Autoencoder.
    Encoder has convolutional layers and max pool layers.

    Decoder has trnanspose convolutional layers and
    upsampling layers.

    Args:
        seq_len: length of input sequence
        in_channels: Number of channels in the input
        enc_channels: Number of channels (filters) in each layer of the encoder
        enc_kernel_sizes: Size of convolutional kernel for each layer
                          Can be both a tuple/list or an integer.
                          If integer, then the same kernel size is applied to all layers.
        pool_kernel_size: Kernel size of the maxpool layer (encoder)
                          and upsample layer (decoder)
        dec_activation: The final activation for the decoder
                        Supported values include: ("sigmoid", "tanh", "relu")
                        If None then no output activation is added

    Note: Length of list/tuple of enc_channels and enc_kernel_sizes must be equal
    """

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        enc_channels: Sequence[int] = (16, 8),
        enc_kernel_sizes: Union[int, Sequence[int]] = 3,
        pool_kernel_size: int = 2,
        dec_activation: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.in_channels = in_channels

        if isinstance(enc_kernel_sizes, int):
            enc_kernel_sizes = [enc_kernel_sizes for _ in range(len(enc_channels))]

        elif isinstance(enc_kernel_sizes, (tuple, list)):
            assert len(enc_channels) == len(
                enc_kernel_sizes
            ), "enc_channels and enc_kernel_sizes should be of the same length"
        else:
            raise TypeError(f"Invalid enc_kernel_sizes type provided: {enc_kernel_sizes}")

        self.encoder = Encoder(
            num_channels=enc_channels,
            kernel_sizes=enc_kernel_sizes,
            pool_kernel_size=pool_kernel_size,
        )

        dec_channels = list(reversed(enc_channels[:-1]))
        dec_channels.append(in_channels)

        dec_kernel_sizes = list(reversed(enc_kernel_sizes))
        self.decoder = Decoder(
            num_channels=dec_channels,
            kernel_sizes=dec_kernel_sizes,
            upsample_scale_factor=pool_kernel_size,
            final_activation=dec_activation,
        )

        # Do a dry run to initialize lazy modules
        self.forward(torch.rand(1, self.seq_len, self.in_channels))
        self._init_weights()

    def _init_weights(self) -> None:
        r"""
        Initiate parameters in the convolutional
        and transpose convolutional layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_normal_(module.weight, gain=calculate_gain("relu"))

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        batch = self.configure_shape(batch)
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def configure_shape(self, batch: Tensor) -> Tensor:
        return batch.view(-1, self.in_channels, self.seq_len)

    def encode(self, batch: Tensor) -> Tensor:
        batch = self.configure_shape(batch)
        return self.encoder(batch)

    def _get_reconstruction_loss(self, batch: Tensor) -> Tensor:
        _, recon = self.forward(batch)
        x = batch.view(-1, self.in_channels, self.seq_len)
        return self.criterion(x, recon)

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Returns reconstruction for streaming input"""
        recon = self.reconstruction(batch)
        recon = recon.view(-1, self.seq_len, self.in_channels)
        return self.criterion(batch, recon, reduction="none")


class SparseConv1dAE(Conv1dAE):
    r"""
    Sparse Autoencoder for a Conv1d network.
    It inherits from VanillaAE class and serves as a wrapper around base network models.
    Sparse Autoencoder is a type of autoencoder that applies sparsity constraint.
    This helps in achieving information bottleneck even when the number of hidden units is huge.
    It penalizes the loss function such that only some neurons are activated at a time.
    This sparsity penalty helps in preventing overfitting.
    More details about Sparse Autoencoder can be found at
        <https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf>

    Args:
        beta: regularization parameter (Defaults to 1e-3)
        rho: sparsity parameter value (Defaults to 0.05)
        **kwargs: VanillaAE kwargs
    """

    def __init__(self, beta=1e-3, rho=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.rho = rho

    def kl_divergence(self, activations: Tensor) -> Tensor:
        r"""
        Loss function for computing sparse penalty based on KL (Kullback-Leibler) Divergence.
        KL Divergence measures the difference between two probability distributions.

        Args:
            activations: encoded output from the model layer-wise

        Returns:
            Tensor
        """
        rho_hat = torch.mean(activations, dim=0)
        rho = torch.full(rho_hat.size(), self.rho, device=self.device)
        kl_loss = nn.KLDivLoss(reduction="sum")
        _dim = 0 if rho_hat.dim() == 1 else 1
        return kl_loss(torch.log_softmax(rho_hat, dim=_dim), torch.softmax(rho, dim=_dim))

    def _get_reconstruction_loss(self, batch):
        latent, recon = self.forward(batch)
        batch = batch.view(-1, self.in_channels, self.seq_len)
        loss = self.criterion(batch, recon)
        penalty = self.kl_divergence(latent)
        return loss + penalty
