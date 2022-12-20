import logging

import numpy as np
from numpy.typing import ArrayLike

from numalogic.tools import DataIndependentTransformers

LOGGER = logging.getLogger(__name__)


class LogTransformer(DataIndependentTransformers):
    def __init__(self, add_factor=2):
        self.add_factor = add_factor

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        return np.log(X + self.add_factor)

    def inverse_transform(self, X) -> ArrayLike:
        return np.exp(X) - self.add_factor


class StaticPowerTransformer(DataIndependentTransformers):
    def __init__(self, n: float, add_factor=0):
        self.add_factor = add_factor
        self.n = n

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        return np.power(X + self.add_factor, self.n)

    def inverse_transform(self, X) -> ArrayLike:
        return np.power(X, 1.0 / self.n) - self.add_factor
