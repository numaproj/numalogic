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

from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from numalogic.base import StatelessTransformer


class LogTransformer(StatelessTransformer):
    """
    Applies column-wise log normalization.

    Args:
    ----
        add_factor: float value to be added to the feature before taking log.
    """

    __slots__ = ("add_factor",)

    def __init__(self, add_factor=2):
        self.add_factor = add_factor

    def transform(self, x: npt.NDArray[float], **__) -> npt.NDArray[float]:
        return np.log(x + self.add_factor)

    def inverse_transform(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        return np.exp(x) - self.add_factor


class StaticPowerTransformer(StatelessTransformer):
    """
    Applies column-wise power transformation.

    Args:
    ----
        n: float value to be used as the power.
        add_factor: float value to be added to the feature before taking power.
    """

    def __init__(self, n: float, add_factor=0):
        self.add_factor = add_factor
        self.n = n

    def transform(self, X, **__):
        return np.power(X + self.add_factor, self.n)

    def inverse_transform(self, X) -> npt.NDArray[float]:
        return np.power(X, 1.0 / self.n) - self.add_factor


class DataClipper(StatelessTransformer):
    """
    Applies column-wise ceiling transformation.

    Args:
    ----
        lower: lower bound for clipping.
        upper: upper bound for clipping.
    """

    __slots__ = ("lower", "upper")

    def __init__(
        self,
        lower: Optional[Sequence[str]] = None,
        upper: Optional[Sequence[str]] = None,
    ):
        self.lower, self.upper = self._validate_args(lower, upper)

    @staticmethod
    def _validate_args(
        lower: Optional[Sequence[str]] = None,
        upper: Optional[Sequence[str]] = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        lower, upper = np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32)
        if np.isnan(lower) and np.isnan(upper):
            raise ValueError("At least one of lower or upper should be provided.")
        if isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray):
            if len(lower) != len(upper):
                raise ValueError("lower and upper should have the same length.")
            if np.any(np.less(upper, lower)):
                raise ValueError("lower value should be less than or equal to upper value")
        return lower, upper

    def transform(self, x: npt.NDArray[float], **__) -> npt.NDArray[float]:
        _df = pd.DataFrame(x, dtype=np.float32)
        if self.upper is not None:
            _df.clip(upper=self.upper, axis=1, inplace=True)
        if self.lower is not None:
            _df.clip(lower=self.lower, axis=1, inplace=True)
        return _df.to_numpy(dtype=np.float32)


class GaussianNoiseAdder(StatelessTransformer):
    """
    Applies Gaussian noise to data.

    Args:
    ----
        scale: small float value to be used as the noise factor (default: 1e-8).
        positive_only: bool value to indicate whether
            to use absolute value of the noise (default: True).
        seed: int value to be used as the random seed (default: 42).
    """

    __slots__ = ("_rng", "_is_abs", "_scale")

    def __init__(self, scale: float = 1e-8, positive_only: bool = True, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._is_abs = positive_only
        self._scale = scale

    def transform(self, x: npt.NDArray[float], **__) -> npt.NDArray[float]:
        noise = self._rng.normal(loc=0.0, scale=self._scale, size=x.shape)
        if self._is_abs:
            noise = np.abs(noise)
        return x + noise


class DifferenceTransform(StatelessTransformer):
    """
    Apply feature wise differencing.

    Note: First value is backfilled with the first non-NAN value.
    """

    def transform(self, input_: npt.NDArray, **__):
        diff_df = pd.DataFrame(input_).diff().bfill()
        return diff_df.to_numpy(dtype=np.float32)


class FlattenVector(StatelessTransformer):
    """A stateless transformer that flattens a vector.

    Args:
    ____
        n_features: number of features

    """

    def __init__(self, n_features: int):
        self.n_features = n_features

    def transform(self, X: npt.NDArray[float], **__) -> npt.NDArray[float]:
        return X.flatten().reshape(-1, 1)

    def inverse_transform(self, X: npt.NDArray[float]) -> npt.NDArray[float]:
        return X.reshape(-1, self.n_features)
