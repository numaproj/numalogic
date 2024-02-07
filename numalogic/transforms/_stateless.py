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

import numpy as np
import numpy.typing as npt

from numalogic.base import StatelessTransformer


class LogTransformer(StatelessTransformer):
    """
    Applies column-wise log normalization.

    Args:
    ----
        add_factor: float value to be added to the feature before taking log.
    """

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


class FlattenVector(StatelessTransformer):
    """A stateless transformer that flattens a vector.

    Args:
    ____
        n_features: number of features

    """

    def __init__(self, n_features: int, seq_length: int):
        self.n_features = n_features
        self.seq_length = seq_length

    def transform(self, X: npt.NDArray[float], seq_length=20) -> npt.NDArray[float]:
        # Calculate the number of elements to take from each column
        n = seq_length // X.shape[1]

        # Initialize an empty list to store the results
        result = []

        # Loop over the array in chunks of size n
        for i in range(0, X.shape[0], n):
            # Loop over the columns of X
            for j in range(X.shape[1]):
                # Take n elements from the current column and append them to the result
                result.extend(X[i : i + n, j])

        # Convert the result to a numpy array with shape (seq_length, 1)
        result = np.array(result).reshape(-1, 1)

        return result

    def inverse_transform(self, X: npt.NDArray[float]) -> npt.NDArray[float]:
        # Todo: Change this aswell. Fix the inverse flatten aswell
        return X.reshape(self.n_features, -1).T
