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
from torch.distributions import kl_divergence, Bernoulli

from numalogic.models.autoencoder.base import BaseAE
from numalogic.tools.exceptions import LayerSizeMismatchError


class _VanillaEncoder(nn.Module):
    r"""Encoder module for the VanillaAE.

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features
        layersizes: encoder layer size
        dropout_p: the dropout value
        batchnorm: Flag to enable/diasable batch normalization
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
        self.input_size = seq_len * n_features
        self.n_features = n_features
        self.dropout_p = dropout_p
        self.bnorm = batchnorm

        layers = self._construct_layers(layersizes)
        self.encoder = nn.Sequential(nn.Flatten(), *layers)

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
        start_layersize = self.input_size

        for lsize in layersizes[:-1]:
            _l = [nn.Linear(start_layersize, lsize)]
            if self.bnorm:
                _l.append(nn.BatchNorm1d(lsize))
            layers.extend([*_l, nn.Tanh(), nn.Dropout(p=self.dropout_p)])
            start_layersize = lsize

        _l = [nn.Linear(start_layersize, layersizes[-1])]
        if self.bnorm:
            _l.append(nn.BatchNorm1d(layersizes[-1]))
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
        batchnorm: flag to enable/disable batch normalization
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
        self.out_size = seq_len * n_features
        self.n_features = n_features
        self.dropout_p = dropout_p
        self.bnorm = batchnorm

        layers = self._construct_layers(layersizes)
        self.decoder = nn.Sequential(*layers, nn.Unflatten(-1, (n_features, seq_len)))

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
            _l = [nn.Linear(layersizes[idx], layersizes[idx + 1])]
            if self.bnorm:
                _l.append(nn.BatchNorm1d(layersizes[idx + 1]))
            layers.extend([*_l, nn.Tanh(), nn.Dropout(p=self.dropout_p)])

        layers.append(nn.Linear(layersizes[-1], self.out_size))
        return layers


class VanillaAE(BaseAE):
    r"""Vanilla Autoencoder model comprising Fully connected layers only.

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features
        encoder_layersizes: encoder layer size (default = Sequence[int] = (16, 8))
        decoder_layersizes: decoder layer size (default = Sequence[int] = (8, 16))
        dropout_p: the dropout value (default=0.25)
        batchnorm: Flag to enable batch normalization (default=False)
        **kwargs: BaseAE kwargs
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int = 1,
        encoder_layersizes: Sequence[int] = (16, 8),
        decoder_layersizes: Sequence[int] = (8, 16),
        dropout_p: float = 0.25,
        batchnorm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.dropout_prob = dropout_p
        self.n_features = n_features

        if encoder_layersizes[-1] != decoder_layersizes[0]:
            raise LayerSizeMismatchError(
                f"Last layersize of encoder: {encoder_layersizes[-1]} "
                f"does not match first layersize of decoder: {decoder_layersizes[0]}"
            )

        self.encoder = _VanillaEncoder(
            seq_len=seq_len,
            n_features=n_features,
            layersizes=encoder_layersizes,
            dropout_p=dropout_p,
            batchnorm=batchnorm,
        )
        self.decoder = _Decoder(
            seq_len=seq_len,
            n_features=n_features,
            layersizes=decoder_layersizes,
            dropout_p=dropout_p,
            batchnorm=batchnorm,
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize the parameters in the model."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        batch = torch.swapdims(batch, 1, 2)
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return encoded, torch.swapdims(decoded, 1, 2)

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        """Returns reconstruction for streaming input."""
        recon = self.reconstruction(batch)
        return self.criterion(batch, recon, reduction="none")


class MultichannelAE(BaseAE):
    r"""Multichannel Vanilla Autoencoder model based on the vanilla encoder and decoder.
        Each channel is an isolated neural network.

    Args:
    ----
        seq_len: sequence length / window length
        n_channels: num of channels, each channel is a separate neural network
        encoder_layersizes: encoder layer size (default = Sequence[int] = (16, 8))
        decoder_layersizes: decoder layer size (default = Sequence[int] = (8, 16))
        dropout_p: the dropout value (default=0.25)
        batchnorm: Flag to enable batch normalization (default=False)
        **kwargs: BaseAE kwargs
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
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
        # The number of features per channel default to 1 in this architecture
        self.n_features = 1

        if encoder_layersizes[-1] != decoder_layersizes[0]:
            raise LayerSizeMismatchError(
                f"Last layersize of encoder: {encoder_layersizes[-1]} "
                f"does not match first layersize of decoder: {decoder_layersizes[0]}"
            )

        for i in range(self.n_channels):
            encoder = _VanillaEncoder(
                seq_len=seq_len,
                n_features=self.n_features,
                layersizes=encoder_layersizes,
                dropout_p=dropout_p,
                batchnorm=batchnorm,
            )
            decoder = _Decoder(
                seq_len=seq_len,
                n_features=self.n_features,
                layersizes=decoder_layersizes,
                dropout_p=dropout_p,
                batchnorm=batchnorm,
            )

            encoder.apply(self.init_weights)
            decoder.apply(self.init_weights)
            setattr(self, f"channel_encoder{i}", encoder)
            setattr(self, f"channel_decoder{i}", decoder)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize the parameters in the model."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        encoded_all, decoded_all = [], []
        batch = torch.swapdims(batch, 1, 2)

        for i in range(self.n_channels):
            encoder = getattr(self, f"channel_encoder{i}")
            decoder = getattr(self, f"channel_decoder{i}")

            batch_channel = batch[:, [i]]

            encoded = encoder(batch_channel)
            decoded = decoder(encoded)

            encoded_all.append(encoded)
            decoded_all.append(decoded)

        encoded_all = torch.stack(encoded_all, dim=-1)
        encoded_all = torch.squeeze(encoded_all, 1)

        decoded_all = torch.stack(decoded_all, dim=-1)
        decoded_all = torch.squeeze(decoded_all, 1)

        return encoded_all, decoded_all

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        """Returns reconstruction for streaming input."""
        recon = self.reconstruction(batch)
        return self.criterion(batch, recon, reduction="none")


class _SparseVanillaEncoder(_VanillaEncoder):
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
        start_layersize = self.input_size

        for lsize in layersizes[:-1]:
            _l = [nn.Linear(start_layersize, lsize)]
            if self.bnorm:
                _l.append(nn.BatchNorm1d(lsize))
            layers.extend([*_l, nn.Tanh(), nn.Dropout(p=self.dropout_p)])
            start_layersize = lsize

        _l = [nn.Linear(start_layersize, layersizes[-1])]
        if self.bnorm:
            _l.append(nn.BatchNorm1d(layersizes[-1]))
        layers.extend([*_l, nn.ReLU()])

        return layers


class SparseVanillaAE(VanillaAE):
    r"""Sparse Autoencoder for a fully connected network.
    It inherits from VanillaAE class and serves as a wrapper around base network models.
    Sparse Autoencoder is a type of autoencoder that applies sparsity constraint.
    This helps in achieving information bottleneck even when the number of hidden units is huge.
    It penalizes the loss function such that only some neurons are activated at a time.
    This sparsity penalty helps in preventing overfitting.
    More details about Sparse Autoencoder can be found at
        <https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf>.

    Args:
    ----
        seq_len: sequence length / window length
        n_features: num of features
        encoder_layersizes: encoder layer size (default = Sequence[int] = (16, 8))
        decoder_layersizes: decoder layer size (default = Sequence[int] = (8, 16))
        dropout_p: the dropout value (default=0.25)
        batchnorm: Flag to enable batch normalization (default=False)
        beta: Regularization factor (Defaults to 1e-3)
        rho: Sparsity parameter value (Defaults to 0.05)
        **kwargs: BaseAE kwargs
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int = 1,
        encoder_layersizes: Sequence[int] = (16, 8),
        decoder_layersizes: Sequence[int] = (8, 16),
        dropout_p: float = 0.25,
        batchnorm: bool = True,
        beta=1e-3,
        rho=0.05,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            n_features=n_features,
            encoder_layersizes=encoder_layersizes,
            decoder_layersizes=decoder_layersizes,
            dropout_p=dropout_p,
            batchnorm=batchnorm,
            **kwargs,
        )
        self.encoder = _SparseVanillaEncoder(
            seq_len=seq_len,
            n_features=n_features,
            layersizes=encoder_layersizes,
            dropout_p=dropout_p,
            batchnorm=batchnorm,
        )
        self.beta = beta
        self.rho = rho

    def kl_divergence(self, activations: Tensor) -> Tensor:
        r"""Loss function for computing sparse penalty based on KL (Kullback-Leibler) Divergence.
        KL Divergence measures the difference between two probability distributions.

        Args:
        ----
            activations: encoded output from the model layer-wise

        Returns
        -------
            Tensor
        """
        rho_hat = torch.mean(activations, dim=0)
        rho = torch.full(rho_hat.size(), self.rho, device=self.device)
        kl_loss = kl_divergence(
            Bernoulli(logits=torch.log(rho)), Bernoulli(logits=torch.log(rho_hat))
        )
        return torch.sum(torch.clamp(kl_loss, max=1.0))

    def get_reconstruction_loss(self, batch: Tensor, reduction="mean") -> Tensor:
        latent, recon = self.forward(batch)
        loss = self.criterion(batch, recon, reduction=reduction)
        penalty = self.kl_divergence(latent)
        return loss + (self.beta * penalty)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        recon = self.reconstruction(batch)
        loss = self.criterion(batch, recon)
        self.log("val_loss", loss)
        return loss
