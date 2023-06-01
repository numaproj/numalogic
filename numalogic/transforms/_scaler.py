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


import logging

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from numalogic.base import BaseTransformer

LOGGER = logging.getLogger(__name__)


class TanhScaler(BaseTransformer):
    r"""Tanh Estimator applies column-wise tanh normalization to the Z-score,
    and scales the values between 0 and 1.

    After scaling, the data has a mean of 0.5.

    The coeff parameter determines the spread of the scores.
    Higher the value, the linear portion of the curve will have a higher slope
    but will reach the asymptote (flatten out) earlier.

    Args:
    ----
        coeff: float value determining the spread of the scores
        eps: minimum value below which the feature will be treated as constant.
             In order to avoid division by zero or a very small number,
             standard deviation will be set as 1 for that feature.

    References
    ----------
        Nandakumar, Jain, Ross. 2005. Score Normalization in
        Multimodal Biometric Systems, Pattern Recognition 38, 2270-2285.
        https://web.cse.msu.edu/~rossarun/pubs/RossScoreNormalization_PR05.pdf
    """

    __slots__ = ("_coeff", "_std", "_mean", "_eps")

    def __init__(self, coeff: float = 0.2, eps: float = 1e-10):
        self._coeff = coeff
        self._std = None
        self._mean = None
        self._eps = eps

    def fit(self, x: npt.NDArray[float]) -> Self:
        self._mean = np.mean(x, axis=0)
        self._std = np.std(x, axis=0)
        self._check_if_constant(x)
        return self

    def transform(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        x_std_scaled = (x - self._mean) / self._std
        return 0.5 * (np.tanh(self._coeff * x_std_scaled) + 1)

    def fit_transform(self, x: npt.NDArray[float], y=None, **_) -> npt.NDArray[float]:
        return self.fit(x).transform(x)

    def _check_if_constant(self, x: npt.NDArray[float]) -> None:
        delta = np.max(x, axis=0) - np.min(x, axis=0)
        self._std[delta < self._eps] = 1.0
