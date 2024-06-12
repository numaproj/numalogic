from collections.abc import Sequence

import torch
from torch import nn, Tensor

from numalogic.models.autoencoder.base import BaseAE
from numalogic.tools.exceptions import LayerSizeMismatchError
from numalogic.tools.layer import IndependentChannelLinear


class _VanillaEncoder(nn.Module):
    r"""Encoder module for the VanillaAE.

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features
        layersizes: encoder layer size
        dropout_p: the dropout value

    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        layersizes: Sequence[int],
        dropout_p: float,
        batchnorm: bool,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.dropout_p = dropout_p
        self.bnorm = batchnorm

        layers = self._construct_layers(layersizes)
        self.encoder = nn.Sequential(*layers)

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

        for lsize in layersizes[:-1]:
            _l = [IndependentChannelLinear(start_layersize, lsize, self.n_features)]
            if self.bnorm:
                _l.append(nn.BatchNorm1d(self.n_features))
            layers.extend([*_l, nn.Tanh(), nn.Dropout(p=self.dropout_p)])
            start_layersize = lsize

        _l = [IndependentChannelLinear(start_layersize, layersizes[-1], self.n_features)]
        if self.bnorm:
            _l.append(nn.BatchNorm1d(self.n_features))
        layers.extend([*_l, nn.Tanh(), nn.Dropout(p=self.dropout_p)])

        return layers

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class _Decoder(nn.Module):
    r"""Decoder module for the autoencoder module.

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features
        layersizes: decoder layer size
        dropout_p: the dropout value

    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        layersizes: Sequence[int],
        dropout_p: float,
        batchnorm: bool,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.dropout_p = dropout_p
        self.bnorm = batchnorm

        layers = self._construct_layers(layersizes)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def _construct_layers(self, layersizes: Sequence[int]) -> nn.ModuleList:
        r"""Utility function to generate a simple feedforward network layer.

        Args:
        ----
            layersizes: layer size

        Returns
        -------
            A simple feedforward network layer
        """
        layers = nn.ModuleList()

        for idx, _ in enumerate(layersizes[:-1]):
            _l = [IndependentChannelLinear(layersizes[idx], layersizes[idx + 1], self.n_features)]
            if self.bnorm:
                _l.append(nn.BatchNorm1d(self.n_features))
            layers.extend([*_l, nn.Tanh(), nn.Dropout(p=self.dropout_p)])

        layers.append(IndependentChannelLinear(layersizes[-1], self.seq_len, self.n_features))
        return layers


class VanillaICAE(BaseAE):
    r"""Vanilla Autoencoder model with Independent isolated Channels based
    on the vanilla encoder and decoder. Each channel is an isolated neural network.

    Args:
    ----
        seq_len: sequence length / window length
        n_channels: num of features/channel, each channel is a separate neural network
        encoder_layersizes: encoder layer size (default = Sequence[int] = (16, 8))
        decoder_layersizes: decoder layer size (default = Sequence[int] = (8, 16))
        dropout_p: the dropout value (default=0.25)
        batchnorm: Flag to enable batch normalization (default=False)
        **kwargs: BaseAE kwargs
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int = 1,
        encoder_layersizes: Sequence[int] = (16, 8),
        decoder_layersizes: Sequence[int] = (8, 16),
        dropout_p: float = 0.25,
        batchnorm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.dropout_prob = dropout_p
        self.n_channels = n_channels

        if encoder_layersizes[-1] != decoder_layersizes[0]:
            raise LayerSizeMismatchError(
                f"Last layersize of encoder: {encoder_layersizes[-1]} "
                f"does not match first layersize of decoder: {decoder_layersizes[0]}"
            )

        self.encoder = _VanillaEncoder(
            seq_len=seq_len,
            n_features=n_channels,
            layersizes=encoder_layersizes,
            dropout_p=dropout_p,
            batchnorm=batchnorm,
        )
        self.decoder = _Decoder(
            seq_len=seq_len,
            n_features=n_channels,
            layersizes=decoder_layersizes,
            dropout_p=dropout_p,
            batchnorm=batchnorm,
        )

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        batch = torch.swapdims(batch, 1, 2)
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return encoded, torch.swapdims(decoded, 1, 2)

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        """Returns reconstruction for streaming input."""
        recon = self.reconstruction(batch)
        return self.criterion(batch, recon, reduction="none")
