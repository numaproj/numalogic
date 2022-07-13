import numpy as np
from numpy.typing import ArrayLike


def tanh_norm(scores: ArrayLike, scale_factor=10, smooth_factor=10) -> ArrayLike:
    return scale_factor * np.tanh(scores / smooth_factor)
