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
