from collections.abc import Sequence, Callable
from typing import Final

import torch
from torch import nn, Tensor, optim
from torch.distributions import MultivariateNormal, kl_divergence
import torch.nn.functional as F
import pytorch_lightning as pl

from numalogic.tools.exceptions import ModelInitializationError


_DEFAULT_KERNEL_SIZE: Final[int] = 3
_DEFAULT_STRIDE: Final[int] = 2


class CausalConv1d(nn.Conv1d):
    """Temporal convolutional layer with causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.__padding, 0)))


class CausalConvBlock(nn.Module):
    """Basic convolutional block consisting of:
    - causal 1D convolutional layer
    - batch norm
    - relu activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_: Tensor) -> Tensor:
        return self.relu(self.bnorm(self.conv(input_)))


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
        num_samples: int = 10,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.nsamples = num_samples

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
        out = out.view(-1, self.seq_len, self.n_features)
        return self.td_linear(out)


def _init_criterion(loss_fn: str) -> Callable:
    if loss_fn == "huber":
        return F.huber_loss
    if loss_fn == "l1":
        return F.l1_loss
    if loss_fn == "mse":
        return F.mse_loss
    raise ValueError(f"Unsupported loss function provided: {loss_fn}")


class Conv1dVAE(pl.LightningModule):
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
        conv_channels: number of convolutional channels
        latent_dim: latent dimension
        num_samples: number of samples to draw from the latent distribution
        lr: learning rate
        weight_decay: weight decay factor for regularization
        loss_fn: loss function used to train the model
                    supported values include: {huber, l1, mae}
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        latent_dim: int,
        conv_channels: Sequence[int] = (16,),
        num_samples: int = 10,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        loss_fn: str = "mse",
    ):
        super().__init__()
        self._lr = lr
        self.weight_decay = weight_decay
        self.criterion = _init_criterion(loss_fn)

        self._total_kld_loss = 0.0
        self._total_train_loss = 0.0
        self._total_val_loss = 0.0

        self.seq_len = seq_len
        self.z_dim = latent_dim
        self.n_features = n_features
        self.nsamples = num_samples

        self.encoder = Encoder(
            seq_len=seq_len,
            n_features=n_features,
            conv_channels=conv_channels,
            latent_dim=latent_dim,
            num_samples=num_samples,
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
        samples = p.rsample(sample_shape=torch.Size([self.nsamples]))
        z = torch.mean(samples, dim=0)
        x_recon = self.decoder(z)
        return p, x_recon

    @property
    def total_kld_loss(self) -> float:
        return self._total_kld_loss

    @property
    def total_train_loss(self) -> float:
        return self._total_train_loss

    @property
    def total_val_loss(self) -> float:
        return self._total_val_loss

    def reset_train_loss(self) -> None:
        self._total_kld_loss = 0.0
        self._total_train_loss = 0.0

    def reset_val_loss(self) -> None:
        self._total_val_loss = 0.0

    def configure_shape(self, x: Tensor) -> Tensor:
        """Method to configure the batch shape for each type of model architecture."""
        return x.view(-1, self.n_features, self.seq_len)

    def configure_optimizers(self) -> dict:
        optimizer = optim.Adam(self.parameters(), lr=self._lr, weight_decay=self.weight_decay)
        return {"optimizer": optimizer}

    def kld_loss(self, p):
        q = MultivariateNormal(torch.zeros(self.z_dim), torch.eye(self.z_dim))  # True distribution
        kld = kl_divergence(q, p)
        return kld.sum()

    def recon_loss(self, batch: Tensor, recon: Tensor, reduction: str = "sum"):
        return self.criterion(batch, recon, reduction=reduction)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        p, recon = self.forward(batch)
        kld_loss = self.kld_loss(p)
        loss = kld_loss + self.recon_loss(batch, recon)
        self._total_kld_loss += kld_loss.item()
        self._total_train_loss += loss.item()
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        p, recon = self.forward(batch)
        loss = self.recon_loss(batch, recon)
        self._total_val_loss += loss.item()
        return loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        p, recon = self.forward(batch)
        return self.recon_loss(batch, recon, reduction="none")
