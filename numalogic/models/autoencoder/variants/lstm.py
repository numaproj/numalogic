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
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import calculate_gain

from numalogic.models.autoencoder.base import BaseAE

_LOGGER = logging.getLogger(__name__)


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


class LSTMAE(BaseAE):
    r"""
    Long Short-Term Memory (LSTM) based autoencoder.

    Args:
        seq_len: sequence length / window length,
        no_features: number of features
        embedding_dim: embedding dimension for the network
        encoder_layers: number of encoder layers (default = 1)
        decoder_layers: number of decoder layers (default = 1)
        kwargs: BaseAE kwargs
    """

    def __init__(
        self,
        seq_len: int,
        no_features: int,
        embedding_dim: int,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim

        self.encoder = _Encoder(
            seq_len=self.seq_len,
            no_features=self.no_features,
            embedding_size=self.embedding_dim,
            num_layers=encoder_layers,
        )
        self.encoder.apply(self.init_weights)

        self.decoder = _Decoder(
            seq_len=self.seq_len,
            no_features=self.embedding_dim,
            output_size=self.no_features,
            hidden_size=embedding_dim,
            num_layers=decoder_layers,
        )
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        """Returns reconstruction for streaming input"""
        recon = self.reconstruction(batch)
        recon_err = self.criterion(batch, recon, reduction="none")
        return recon_err


class SparseLSTMAE(LSTMAE):
    r"""
    Sparse Autoencoder for an LSTM network.
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
        loss = self.criterion(batch, recon)
        penalty = self.kl_divergence(latent)
        return loss + penalty
