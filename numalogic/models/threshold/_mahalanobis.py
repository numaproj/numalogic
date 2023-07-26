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

from typing import Final

import numpy as np
import numpy.typing as npt

from numalogic.base import BaseThresholdModel
from typing_extensions import Self

from numalogic.tools.exceptions import ModelInitializationError, InvalidDataShapeError

_INLIER: Final[int] = 0
_OUTLIER: Final[int] = 1
_INPUT_DIMS: Final[int] = 2


class MahalanobisThreshold(BaseThresholdModel):
    """
    Multivariate threshold estimator using Mahalanobis distance.

    The maximum outlier probability is used to calculate the k value
    using Chebyshev's inequality. This basically means that the
    probability of values lying outside the interval
    (mean - k * std, mean + k * std) is not more than max_outlier_prob.
    A value of 0.1 means that 90% of the values lie within the interval and
    10% of the values lie outside the interval.

    The threshold is calculated as the
    mean of Mahalanobis distance plus k times the standard deviation
    of Mahalanobis distance.

    Args:
    ----
        max_outlier_prob: maximum outlier probability (default: 0.1)

    Raises
    ------
        ValueError: if max_outlier_prob is not in range (0, 1)

    >>> import numpy as np
    >>> from numalogic.models.threshold import MahalanobisThreshold
    >>> rng = np.random.default_rng(42)
    ...
    >>> x_train = rng.normal(size=(100, 15))
    >>> x_test = rng.normal(size=(30, 15))
    ...
    >>> clf = MahalanobisThreshold()
    >>> clf.fit(x_train)
    ...
    >>> y_test = clf.predict(x_test)
    >>> test_scores = clf.score_samples(x_test)
    """

    def __init__(self, max_outlier_prob: float = 0.1):
        if not 0.0 < max_outlier_prob < 1.0:
            raise ValueError("max_outlier_prob should be in range (0, 1)")
        self._k = self._chebyshev_k(max_outlier_prob)
        self._distr_mean = None
        self._cov_inv = None
        self._md_thresh = None
        self._is_fitted = False

    @property
    def threshold(self) -> float:
        """Returns the threshold value."""
        return self._md_thresh

    @property
    def std_factor(self) -> float:
        """Returns the k value calculated using Chebyshev's inequality."""
        return self._k

    @staticmethod
    def _chebyshev_k(p: float) -> float:
        """Calculate the k value using Chebyshev's inequality."""
        return np.reciprocal(np.sqrt(p))

    @staticmethod
    def _validate_input(x: npt.NDArray[float]) -> None:
        """Validate the input matrix shape."""
        if x.ndim != _INPUT_DIMS:
            raise InvalidDataShapeError(f"Input matrix should have 2 dims, given shape: {x.shape}.")

    def fit(self, x: npt.NDArray[float]) -> Self:
        """
        Fit the estimator on the training set.

        Args:
        ----
            x: training data of shape (n_samples, n_features)

        Returns
        -------
            self

        Raises
        ------
            InvalidDataShapeError: if the input matrix is not 2D
        """
        self._validate_input(x)
        self._distr_mean = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False)
        self._cov_inv = np.linalg.pinv(cov)
        mahal_dist = self.mahalanobis(x)
        self._md_thresh = np.mean(mahal_dist) + self._k * np.std(mahal_dist)
        self._is_fitted = True
        return self

    def mahalanobis(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Calculate the Mahalanobis distance.

        Args:
        ----
            x: input data of shape (n_samples, n_features)

        Returns
        -------
            Mahalanobis distance vector of shape (n_samples,)
        """
        x_distance = x - self._distr_mean
        mahal_grid = x_distance @ self._cov_inv @ x_distance.T
        return np.sqrt(np.diagonal(mahal_grid))

    def predict(self, x: npt.NDArray[float]) -> npt.NDArray[int]:
        """
        Returns an integer array of same shape as input.
        0 denotes inlier, 1 denotes outlier.

        Args:
        ----
            x: input data of shape (n_samples, n_features)

        Returns
        -------
            Integer Array of shape (n_samples,)

        Raises
        ------
            ModelInitializationError: if the model is not fitted yet
            InvalidDataShapeError: if the input matrix is not 2D
        """
        if not self._is_fitted:
            raise ModelInitializationError("Model not fitted yet.")
        self._validate_input(x)
        md = self.mahalanobis(x)
        y_hat = np.zeros(x.shape[0], dtype=int)
        y_hat[md >= self._md_thresh] = _OUTLIER
        return y_hat

    def score_samples(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Returns the outlier score for each sample.

        Score is calculated as the ratio of Mahalanobis distance to threshold.
        A score greater than 1.0 means that the sample is an outlier.

        Args:
        ----
            x: input data of shape (n_samples, n_features)

        Returns
        -------
            Outlier score vector of shape (n_samples,)

        Raises
        ------
            RuntimeError: if the model is not fitted yet
            InvalidDataShapeError: if the input matrix is not 2D
        """
        if not self._is_fitted:
            raise ModelInitializationError("Model not fitted yet.")
        self._validate_input(x)
        return self.mahalanobis(x) / self._md_thresh
