import logging
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.init import calculate_gain

from numalogic.models.autoencoder.base import BaseAE

LOGGER = logging.getLogger(__name__)


class Conv1dAE(BaseAE):
    r"""
    One dimensional Convolutional Autoencoder with multichannel support.

    Args:
        seq_len: length of input sequence
        in_channels: Number of channels in the input
        enc_channels: Number of channels produced by the convolution
        kernel_size: kernel size (default=7)
        stride: stride length (default=2)
        padding: padding parameter for encoder (default=3)
        output_padding: padding parameter for decoder (default=1)
    """

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        enc_channels: int,
        kernel_size=7,
        stride=2,
        padding=3,
        output_padding=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels, enc_channels, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm1d(enc_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                enc_channels,
                enc_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm1d(enc_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(enc_channels, in_channels, kernel_size=7, padding=3),
            nn.Upsample(scale_factor=2),
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        r"""
        Initiate parameters in the transformer model.
        """
        if type(m) in (nn.ConvTranspose1d, nn.Conv1d):
            nn.init.xavier_normal_(m.weight, gain=calculate_gain("relu"))

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        batch = batch.view(-1, self.in_channels, self.seq_len)
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def _get_reconstruction_loss(self, batch):
        _, recon = self.forward(batch)
        x = batch.view(-1, self.in_channels, self.seq_len)
        return self.criterion(x, recon)

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        """Returns reconstruction for streaming input"""
        recon = self.reconstruction(batch)
        recon = recon.view(-1, self.seq_len, self.in_channels)
        recon_err = self.criterion(batch, recon, reduction="none")
        return recon_err


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
        rho = torch.full(rho_hat.size(), self.rho)
        kl_loss = nn.KLDivLoss(reduction="sum")
        _dim = 0 if rho_hat.dim() == 1 else 1
        return kl_loss(torch.log_softmax(rho_hat, dim=_dim), torch.softmax(rho, dim=_dim))

    def _get_reconstruction_loss(self, batch):
        latent, recon = self.forward(batch)
        batch = batch.view(-1, self.in_channels, self.seq_len)
        loss = self.criterion(batch, recon)
        penalty = self.kl_divergence(latent)
        return loss + penalty
