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
from typing import Optional, Union

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
        lower: Optional[Union[float, Sequence[float]]] = None,
        upper: Optional[Union[float, Sequence[float]]] = None,
    ):
        self.lower, self.upper = self._validate_args(lower, upper)

    @staticmethod
    def _validate_args(
        lower: Optional[Union[float, Sequence[float]]],
        upper: Optional[Union[float, Sequence[float]]],
    ) -> Optional[tuple[Optional[Union[float, npt.NDArray]], Optional[Union[float, npt.NDArray]]]]:
        if lower is None and upper is None:
            raise ValueError("At least one of lower or upper should be provided.")
        if isinstance(lower, Sequence) and isinstance(upper, Sequence):
            if len(lower) != len(upper):
                raise ValueError("lower and upper should have the same length.")
            lower, upper = np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)
        elif lower is not None and upper is not None:
            if type(lower) is not type(upper):
                if isinstance(lower, Sequence):
                    upper = np.asarray(upper, dtype=np.float32)
                else:
                    lower = np.asarray(lower, dtype=np.float32)
        if upper is not None and lower is not None and np.any(lower > upper):
            raise ValueError("lower value should be less than or equal to upper value")
        return lower, upper

    def transform(self, x: npt.NDArray[float], **__) -> npt.NDArray[float]:
        _df = pd.DataFrame(x, dtype=np.float32)
        _df = _df.clip(upper=self.upper, lower=self.lower, axis=1)
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


class FlattenVectorWithPadding(StatelessTransformer):
    """A stateless transformer that flattens some of the columns and does padding on the rest.

    Args:
    ____
        features: list of all feature names in the order of columns of the payload matrix
        flatten_features: list of feature names to be flattened
        padding_with: numerical value to be used for padding, default is 0

    """

    __slots__ = (
        "features",
        "flatten_features",
        "padding_with",
        "padding_features",
        "flatten_indexes",
        "padding_indexes",
    )

    @staticmethod
    def _feature_indexes(features_all: list[str], features: list[str]) -> list[int]:
        return [features_all.index(f) for f in features]

    def __init__(self, features: list[str], flatten_features: list[str], padding_with: float = 0.0):
        self.features = features
        self.flatten_features = flatten_features
        self.padding_with = padding_with

        self.padding_features = list(set(features) - set(flatten_features))
        if not self.padding_features:
            raise ValueError("At least one feature should be left for padding.")
        self.flatten_indexes = self._feature_indexes(features, self.flatten_features)
        self.padding_indexes = self._feature_indexes(features, self.padding_features)

    def transform(self, X: npt.NDArray[float], **__) -> npt.NDArray[float]:
        X_flatten = X[:, self.flatten_indexes].flatten().reshape(-1, 1)
        padding_len = X_flatten.shape[0] - X.shape[0]
        X_padding = np.pad(
            X[:, self.padding_indexes],
            ((0, padding_len), (0, 0)),
            mode="constant",
            constant_values=self.padding_with,
        )
        return np.concatenate([X_flatten, X_padding], axis=1)

    def inverse_transform(self, X: npt.NDArray[float]) -> npt.NDArray[float]:
        X_flatten = X[:, 0].reshape(-1, len(self.flatten_features))
        original_len = X.shape[0] - X_flatten.shape[0]
        X_padding_removed = X[:original_len, 1:]
        return np.concatenate([X_flatten, X_padding_removed], axis=1)
