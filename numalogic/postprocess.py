import numpy as np
from numpy.typing import ArrayLike

from numalogic.tools import DataIndependentTransformers


def tanh_norm(scores: ArrayLike, scale_factor=10, smooth_factor=10) -> ArrayLike:
    return scale_factor * np.tanh(scores / smooth_factor)


class TanhNorm(DataIndependentTransformers):
    def __init__(self, scale_factor=10, smooth_factor=10):
        self.scale_factor = scale_factor
        self.smooth_factor = smooth_factor

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        return tanh_norm(X, scale_factor=self.scale_factor, smooth_factor=self.smooth_factor)
