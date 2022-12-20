from typing import Tuple, Sequence

import torch
from torch import nn, Tensor

from numalogic.models.autoencoder.base import BaseAE
from numalogic.tools.exceptions import LayerSizeMismatchError


class _Encoder(nn.Module):
    r"""
    Encoder module for the autoencoder module.

    Args:
        seq_len: sequence length / window length
        n_features: num of features
        layersizes: encoder layer size
        dropout_p: the dropout value

    """

    def __init__(self, seq_len: int, n_features: int, layersizes: Sequence[int], dropout_p: float):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.dropout_p = dropout_p

        layers = self._construct_layers(layersizes)
        self.encoder = nn.Sequential(*layers)

    def _construct_layers(self, layersizes: Sequence[int]) -> nn.ModuleList:
        r"""
        Utility function to generate a simple feedforward network layer

        Args:
            layersizes: layer size

        Returns:
            A simple feedforward network layer of type nn.ModuleList
        """
        layers = nn.ModuleList()
        start_layersize = self.seq_len

        for lsize in layersizes[:-1]:
            layers.extend(
                [
                    nn.Linear(start_layersize, lsize),
                    nn.BatchNorm1d(self.n_features),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_p),
                ]
            )
            start_layersize = lsize

        layers.extend(
            [
                nn.Linear(start_layersize, layersizes[-1]),
                nn.BatchNorm1d(self.n_features),
                nn.LeakyReLU(),
            ]
        )
        return layers

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class _Decoder(nn.Module):
    r"""
    Decoder module for the autoencoder module.

    Args:
        seq_len: sequence length / window length
        n_features: num of features
        layersizes: decoder layer size
        dropout_p: the dropout value

    """

    def __init__(self, seq_len: int, n_features: int, layersizes: Sequence[int], dropout_p: float):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.dropout_p = dropout_p

        layers = self._construct_layers(layersizes)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def _construct_layers(self, layersizes: Sequence[int]) -> nn.ModuleList:
        r"""
        Utility function to generate a simple feedforward network layer

        Args:
            layersizes: layer size

        Returns:
            A simple feedforward network layer
        """
        layers = nn.ModuleList()

        for idx, _ in enumerate(layersizes[:-1]):
            layers.extend(
                [
                    nn.Linear(layersizes[idx], layersizes[idx + 1]),
                    nn.BatchNorm1d(self.n_features),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_p),
                ]
            )

        layers.append(nn.Linear(layersizes[-1], self.seq_len))
        return layers


class VanillaAE(BaseAE):
    r"""
    Vanilla Autoencoder model comprising Fully connected layers only.

    Args:

        signal_len: sequence length / window length
        n_features: num of features
        encoder_layersizes: encoder layer size (default = Sequence[int] = (16, 8))
        decoder_layersizes: decoder layer size (default = Sequence[int] = (8, 16))
        dropout_p: the dropout value (default=0.25)
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int = 1,
        encoder_layersizes: Sequence[int] = (16, 8),
        decoder_layersizes: Sequence[int] = (8, 16),
        dropout_p: float = 0.25,
        **kwargs,
    ):

        super(VanillaAE, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.dropout_prob = dropout_p
        self.n_features = n_features

        self.save_hyperparameters()

        if encoder_layersizes[-1] != decoder_layersizes[0]:
            raise LayerSizeMismatchError(
                f"Last layersize of encoder: {encoder_layersizes[-1]} "
                f"does not match first layersize of decoder: {decoder_layersizes[0]}"
            )

        self.encoder = _Encoder(
            seq_len=seq_len,
            n_features=n_features,
            layersizes=encoder_layersizes,
            dropout_p=dropout_p,
        )
        self.decoder = _Decoder(
            seq_len=seq_len,
            n_features=n_features,
            layersizes=decoder_layersizes,
            dropout_p=dropout_p,
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        r"""
        Initiate parameters in the transformer model.
        """
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        batch = batch.view(-1, self.n_features, self.seq_len)
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def _get_reconstruction_loss(self, batch):
        _, recon = self.forward(batch)
        x = batch.view(-1, self.n_features, self.seq_len)
        return self.criterion(x, recon)

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        """Returns reconstruction for streaming input"""
        recon = self.reconstruction(batch)
        # batch = batch.view(-1, self.n_features, self.seq_len)
        recon = recon.view(-1, self.seq_len, self.n_features)
        recon_err = self.criterion(batch, recon, reduction="none")
        recon_err = torch.squeeze(recon_err, 0)
        return recon_err


class SparseVanillaAE(VanillaAE):
    r"""
    Sparse Autoencoder for a fully connected network.
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
        x = batch.view(-1, self.n_features, self.seq_len)
        loss = self.criterion(x, recon)
        penalty = self.kl_divergence(latent)
        return loss + penalty
