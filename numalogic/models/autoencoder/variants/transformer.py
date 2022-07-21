from typing import Tuple

import torch
import torch.nn.functional
from torch import nn, Tensor

from numalogic.models.autoencoder.base import TorchAE
from numalogic.preprocess.datasets import SequenceDataset


def _scaled_dot_product(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    r"""
    Calculates scalar_dot_product between three tensors

    Args:
         query: Tensor
         key: Tensor
         value: Tensor
    Returns:
        Tensor
    """

    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = torch.nn.functional.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


def _positional_encoding(
    feature: int,
    seq_len: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    r"""
    Positional Encoding as described in
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        feature: number of features
        seq_len: sequence length
    Returns:
        Tensor
    """
    pos = torch.arange(feature, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e6 ** torch.div(dim, seq_len, rounding_mode="trunc"))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def _feed_forward(dim_input: int = 10, dim_feedforward: int = 2048) -> nn.Module:
    r"""
    Function for creating feedforward network.

    Args:
        dim_input: sequence length / window length (default=1)
        dim_feedforward: the dimension of the feedforward network model (default=2048)

    Returns:
        nn.Module type
    """
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


class _Residual(nn.Module):
    r"""
    Residual Class.

    Args:
        sublayer: feedforward network
        dimension: sequence length / window length
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class _AttentionHead(nn.Module):
    r"""AttentionHead utility class for MultiHeadAttention.

    Args:
        dim_in: Total dimension of the model.
        dim_key: Total number of features for keys.
        dim_query: Total number of features for values.
    """

    def __init__(self, dim_in: int, dim_query: int, dim_key: int):
        super().__init__()
        self.query = nn.Linear(dim_in, dim_query)
        self.key = nn.Linear(dim_in, dim_key)
        self.value = nn.Linear(dim_in, dim_key)

    def forward(self, query, key, value) -> Tensor:
        return _scaled_dot_product(self.query(query), self.key(key), self.value(value))


class MultiHeadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dim_in: Total dimension of the model.
        dim_key: Total number of features for keys.
        dim_query: Total number of features for values.

    """

    def __init__(self, num_heads: int, dim_in: int, dim_query: int, dim_key: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [_AttentionHead(dim_in, dim_query, dim_key) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_key, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(torch.cat([h(query, key, value) for h in self.heads], dim=-1))


class _EncoderLayer(nn.Module):
    r"""EncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need"

    Args:
        dim_model: sequence length / window length (default=1)
        num_heads: the number of heads in the multiheadattention models (default=6)
        dim_feedforward: the dimension of the feedforward network model (default=2048)
        dropout: the dropout value (default=0.1).
    """

    def __init__(
        self,
        dim_model: int = 1,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = _Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = _Residual(
            _feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class Encoder(nn.Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        num_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_model: sequence length / window length (default=1)
        num_heads: the number of heads in the multiheadattention models (default=6)
        dim_feedforward: the dimension of the feedforward network model (default=2048)
        dropout: the dropout value (default=0.1).
    """

    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 1,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):

        super().__init__()
        self.layers = nn.ModuleList(
            [
                _EncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        feature, seq_len = src.size(1), src.size(2)
        src = torch.add(src, _positional_encoding(feature, seq_len))
        for layer in self.layers:
            src = layer(src)
        return src


class _DecoderLayer(nn.Module):
    r"""DecoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need"

    Args:
        dim_model: sequence length / window length (default=1)
        num_heads: the number of heads in the multiheadattention models (default=6)
        dim_feedforward: the dimension of the feedforward network model (default=2048)
        dropout: the dropout value (default=0.1).
    """

    def __init__(
        self,
        dim_model: int = 1,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention_1 = _Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = _Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = _Residual(
            _feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(tgt, memory, memory)
        return self.feed_forward(tgt)


class Decoder(nn.Module):
    r"""Decoder is a stack of N decoder layers

    Args:
        num_layers: the number of sub-decoder-layers in the encoder (default=6).
        dim_model: sequence length / window length (default=1)
        num_heads: the number of heads in the multiheadattention models (default=6)
        dim_feedforward: the dimension of the feedforward network model (default=2048)
        dropout: the dropout value (default=0.1).
    """

    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 1,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        feature, seq_len = tgt.size(1), tgt.size(2)
        tgt = torch.add(tgt, _positional_encoding(feature, seq_len))
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)


class TransformerAE(TorchAE):
    r"""
    Transformer model without masking. Inspiration:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        num_encoder_layers: number of encoder layers in the Encoder (default = 3)
        num_decoder_layers: number of encoder layers in the Decoder (default = 3)
        seq_length: sequence length / window length (default=1)
        num_heads: the number of heads in the multiheadattention models (default=6)
        dim_feedforward: the dimension of the feedforward network model (default=2048)
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. (default: relu)

    """

    def __init__(
        self,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        seq_length: int = 1,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.activation = activation
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            dim_model=seq_length,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            dim_model=seq_length,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        r"""
        Initiate parameters in the transformer model.
        """
        if type(m) in (nn.Linear,):
            nn.init.xavier_uniform_(m.weight, gain=2**0.5)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(x, encoded)
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
