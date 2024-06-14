import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


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


class IndependentChannelLinear(nn.Module):
    """
    Linear layer that treats each feature as independent isolated channels.

    Args:
    ----
        in_features: num of input features
        out_features: num of output features
        n_channels: num of independent channels
        device: device to run on
        dtype: datatype to use
    """

    def __init__(
        self, in_features: int, out_features: int, n_channels: int, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels

        self.weight = nn.Parameter(
            torch.empty((n_channels, in_features, out_features), **factory_kwargs)
        )
        self.bias = nn.Parameter(torch.empty((n_channels, 1, out_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.swapdims(x, 0, 1)
        output = torch.bmm(x, self.weight) + self.bias
        return torch.swapdims(output, 0, 1)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"n_channels={self.n_channels}"
        )
