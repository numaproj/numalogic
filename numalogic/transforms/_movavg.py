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

from numalogic.base import StatelessTransformer
from numalogic.tools.exceptions import InvalidDataShapeError
import numpy.typing as npt


def _allow_only_single_feature(data: npt.NDArray[float]) -> None:
    if data.ndim > 2:
        raise InvalidDataShapeError(
            f"Input data can only be 2 dimensions or less, input size: {data.shape}"
        )
    if data.ndim > 1 and data.shape[1] > 1:
        raise InvalidDataShapeError(
            f"Input data can only have 1 feature column, input shape: {data.shape}"
        )


def expmov_avg_aggregator(
    arr: npt.NDArray[float], beta: float, bias_correction: bool = True
) -> float:
    """Aggregate a window of data into an expoentially weighted moving average value.

    V(n) = (1 - beta) * beta**n * sum(x(i)/beta**i)   [for i = 1 to i = n]

    "1.0 - beta" denotes the weight given to the latest element.

    Args:
    ----
        arr: single feature numpy array
        beta: how much weight to give to the previous weighted average (n-1)th value
        bias_correction: flag to perform bias correction (default: true)

    Raises
    ------
        ValueError: if beta is not between 0 and 1
        InvalidDataShapeError: if input array is not single featured
    """
    if beta <= 0.0 or beta >= 1.0:
        raise ValueError("beta only accepts values between 0 and 1 (not inclusive)")
    _allow_only_single_feature(arr)

    # alpha is the weight given to the latest element
    alpha = 1.0 - beta
    n = len(arr)
    theta = arr.reshape(-1, 1)
    powers = np.arange(n - 1, -1, -1)

    # Calculate decreasing powers of beta of the form
    # [beta**(n-1), beta**(n-2), .., beta**0]
    beta_powers = np.power(beta, powers).reshape(1, -1)

    exp_avg = alpha * (beta_powers @ theta)
    if not bias_correction:
        return exp_avg.item()

    # Perform bias correction
    corrected_exp_avg = exp_avg / (1.0 - np.power(beta, n))
    return corrected_exp_avg.item()


class ExpMovingAverage(StatelessTransformer):
    r"""Calculate the exponential moving averages for a vector.

    This transformation returns an array where each element "n"
    is given by the expression:

    V(n) = (1 - beta) * beta**n * sum(x(i)/beta**i)   [for i = 1 to i = n]

    "1.0 - beta" denotes the weight given to the latest element.

    Without bias correction, early values can tend more towards zero, since V(0) = 0
    Bias correction helps inhibit this issue by dividing with (1 - beta**i)

    Args:
    ----
        beta: how much weight to give to the previous weighted average
        bias_correction: flag to perform bias correction (default: true)

    Note: this only supports single feature input array.

    Raises
    ------
        ValueError: if beta is not between 0 and 1
    """

    __slots__ = ("beta", "bias_correction")

    def __init__(self, beta: float, bias_correction: bool = True):
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError("beta only accepts values between 0 and 1 (not inclusive)")
        self.beta = beta
        self.bias_correction = bias_correction

    def transform(self, input_: npt.NDArray[float], **__):
        r"""Returns transformed output.

        Args:
        ----
            input_: input column vector

        Raises
        ------
            InvalidDataShapeError: if input array is not single featured
        """
        _allow_only_single_feature(input_)

        # alpha is the weight given to the latest element
        alpha = 1.0 - self.beta
        n = len(input_)

        theta = input_.reshape(-1, 1)
        theta_tril = np.multiply(theta.T, np.tril(np.ones((n, n))))
        powers = np.arange(1, n + 1).reshape(-1, 1)

        # Calculate increasing powers of beta of the form,
        # [beta, beta**2, .., beta**n]
        beta_powers = np.power(self.beta, powers)

        # Calculate the array of reciprocals of beta powers of form,
        # [beta**(-1), beta**(-2), .., beta**(-n)]
        beta_arr_inv = np.reciprocal(beta_powers)

        # Calculate the summation of the ratio between (theta(i) / beta**i),
        # [ theta(1)/beta, sum(theta(1)/beta, theta(2)/beta**2), .., ]
        theta_beta_ratio = theta_tril @ beta_arr_inv

        # Elemental multiply with beta powers
        exp_avg = alpha * np.multiply(beta_powers, theta_beta_ratio)
        if not self.bias_correction:
            return exp_avg

        # Calculate array of 1 / (1 - beta**i) values
        return np.divide(exp_avg, 1.0 - beta_powers)
