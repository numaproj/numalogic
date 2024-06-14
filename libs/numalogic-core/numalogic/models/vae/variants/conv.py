from collections.abc import Sequence
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import MultivariateNormal, kl_divergence

from numalogic.models.vae.base import BaseVAE
from numalogic.tools.layer import CausalConvBlock
from numalogic.tools.exceptions import ModelInitializationError

_DEFAULT_KERNEL_SIZE: Final[int] = 3
_DEFAULT_STRIDE: Final[int] = 2


class Encoder(nn.Module):
    """
    Encoder module for Convolutional Variational Autoencoder.

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features
        latent_dim: latent dimension
        conv_channels: number of convolutional channels
        num_samples: number of samples to draw from the latent distribution
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        latent_dim: int,
        conv_channels: Sequence[int] = (16,),
    ):
        super().__init__()

        self.seq_len = seq_len

        conv_layer = CausalConvBlock(
            in_channels=n_features,
            out_channels=conv_channels[0],
            kernel_size=_DEFAULT_KERNEL_SIZE,
            stride=_DEFAULT_STRIDE,
            dilation=1,
        )
        layers = self._construct_conv_layers(conv_channels)
        if layers:
            self.conv_layers = nn.Sequential(conv_layer, *layers)
        else:
            self.conv_layers = conv_layer

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.LazyLinear(latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    @staticmethod
    def _construct_conv_layers(conv_channels) -> nn.ModuleList:
        """Construct dilated causal convolutional layers."""
        layers = nn.ModuleList()
        layer_idx = 1
        while layer_idx < len(conv_channels):
            layers.append(
                CausalConvBlock(
                    conv_channels[layer_idx - 1],
                    conv_channels[layer_idx],
                    kernel_size=_DEFAULT_KERNEL_SIZE,
                    stride=_DEFAULT_STRIDE,
                    dilation=2**layer_idx,
                )
            )
            layer_idx += 1
        return layers

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass returning the mean and
        log variance of the latent distribution.

        Args:
        ----
            x: input tensor of shape (batch_size, n_features, seq_len)

        Returns
        -------
            A tuple of:
                mu: mean of the latent distribution
                logvar: log variance of the latent distribution
        """
        out = self.conv_layers(x)
        out = self.flatten(out)
        out = torch.relu(self.fc(out))
        mu = self.mu(out)
        logvar = F.softplus(self.logvar(out))
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder module for Convolutional Variational Autoencoder.

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features
        num_conv_filters: number of convolutional filters
        latent_dim: latent dimension
    """

    def __init__(self, seq_len: int, n_features: int, num_conv_filters: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.fc = nn.Linear(latent_dim, num_conv_filters * 6)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(num_conv_filters, 6))
        self.conv_tr = nn.ConvTranspose1d(
            in_channels=num_conv_filters,
            out_channels=n_features,
            kernel_size=_DEFAULT_KERNEL_SIZE,
            stride=_DEFAULT_STRIDE,
            padding=1,
            output_padding=1,
        )
        self.bnorm = nn.BatchNorm1d(n_features)
        self.fc_out = nn.LazyLinear(seq_len)
        self.td_linear = nn.Linear(n_features, n_features)

    def forward(self, z: Tensor) -> Tensor:
        out = torch.relu(self.fc(z))
        out = self.unflatten(out)
        out = torch.relu(self.bnorm(self.conv_tr(out)))
        out = torch.relu(self.fc_out(out))
        out = torch.swapdims(out, 1, 2)
        return self.td_linear(out)


class Conv1dVAE(BaseVAE):
    """
    Convolutional Variational Autoencoder for time series data.

    Uses causal convolutions to preserve temporal information in
    the encoded latent space. The decoder non probabilsitc, and
    conists of transposed convolutions and linear layers.

    Note: The model assumes that the input data is of shape
        (batch_size, n_features, seq_len).

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features
        latent_dim: latent dimension
        conv_channels: number of convolutional channels
        beta: disentanglement factor; weightage applied to KLD loss (default=0.1)

    Raises
    ------
        ValueError: if an unsupported loss function is provided
        ModuleInitializationError: if initialization of the model fails due to invalid input,
            invalid hyperparameters, or invalid convolutional kernel size / stride
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        latent_dim: int,
        conv_channels: Sequence[int] = (16,),
        beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.z_dim = latent_dim
        self.n_features = n_features
        self.beta = beta

        self.encoder = Encoder(
            seq_len=seq_len,
            n_features=n_features,
            conv_channels=conv_channels,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            seq_len=seq_len,
            n_features=n_features,
            num_conv_filters=conv_channels[0],
            latent_dim=latent_dim,
        )

        # Do a dry run to initialize lazy modules
        try:
            self.forward(torch.rand(1, seq_len, n_features))
        except (ValueError, RuntimeError) as err:
            raise ModelInitializationError(
                "Model forward pass failed. "
                "Please validate input arguments and the expected input shape "
            ) from err

    def forward(self, x: Tensor) -> tuple[MultivariateNormal, Tensor]:
        x = self.configure_shape(x)
        z_mu, z_logvar = self.encoder(x)
        p = MultivariateNormal(loc=z_mu, covariance_matrix=torch.diag_embed(z_logvar.exp()))
        z = p.rsample()
        x_recon = self.decoder(z)
        return p, x_recon

    def configure_shape(self, x: Tensor) -> Tensor:
        """Method to configure the batch shape for each type of model architecture."""
        return torch.swapdims(x, 1, 2)

    def kld_loss(self, p: MultivariateNormal) -> Tensor:
        """
        Computes the reverse KL divergence between latent distribution and
        the known Multivariate Gaussian prior.

        Args:
        ----
            p: MultivariateNormal distribution

        Returns
        -------
            kld: Reverse KL divergence between p and q
        """
        q = MultivariateNormal(torch.zeros(self.z_dim), torch.eye(self.z_dim))
        kld = kl_divergence(q, p)
        return kld.sum()

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Training step for the model."""
        p, recon = self.forward(batch)
        kld_loss = self.kld_loss(p)
        recon_loss = self.criterion(batch, recon, reduction="sum")
        train_loss = recon_loss + (self.beta * kld_loss)
        self.log_dict(
            {
                "train_loss": train_loss,
                "kld_loss": kld_loss,
                "recon_loss": recon_loss,
            },
            on_epoch=True,
            on_step=False,
        )
        return train_loss
