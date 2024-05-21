from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor


def l2_loss(input_: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute the Torch MSE (L2) loss multiplied with a factor of 0.5."""
    return 0.5 * F.mse_loss(input_, target, reduction=reduction)


def get_loss_fn(loss_fn: str) -> Callable:
    """
    Get the loss function based on the provided loss name.

    Args:
    ----
        loss_fn: loss function name (huber, l1, mse)

    Returns
    -------
        Callable: loss function

    Raises
    ------
        NotImplementedError: If unsupported loss function provided
    """
    if loss_fn == "huber":
        return F.huber_loss
    if loss_fn == "l1":
        return F.l1_loss
    if loss_fn == "mse":
        return l2_loss
    raise NotImplementedError(f"Unsupported loss function provided: {loss_fn}")
