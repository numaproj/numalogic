from typing import Tuple, Sequence

from torch import nn, Tensor

from numalogic.models.autoencoder.base import TorchAE
from numalogic.preprocess.datasets import SequenceDataset
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


class VanillaAE(TorchAE):
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
        signal_len: int,
        n_features: int = 1,
        encoder_layersizes: Sequence[int] = (16, 8),
        decoder_layersizes: Sequence[int] = (8, 16),
        dropout_p: float = 0.25,
    ):

        super(VanillaAE, self).__init__()
        self.seq_len = signal_len
        self.dropout_prob = dropout_p

        if encoder_layersizes[-1] != decoder_layersizes[0]:
            raise LayerSizeMismatchError(
                f"Last layersize of encoder: {encoder_layersizes[-1]} "
                f"does not match first layersize of decoder: {decoder_layersizes[0]}"
            )

        self.encoder = _Encoder(
            seq_len=signal_len,
            n_features=n_features,
            layersizes=encoder_layersizes,
            dropout_p=dropout_p,
        )
        self.decoder = _Decoder(
            seq_len=signal_len,
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
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
        dataset = SequenceDataset(x, __seq_len, permute=True)
        return dataset
