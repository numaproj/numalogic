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
