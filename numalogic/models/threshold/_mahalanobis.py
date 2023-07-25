from typing import Final

import numpy as np
import numpy.typing as npt

from numalogic.base import BaseThresholdModel
from typing_extensions import Self

from numalogic.tools.exceptions import ModelInitializationError

_INLIER: Final[int] = 0
_OUTLIER: Final[int] = 1


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
    def threshold(self):
        """Returns the threshold value."""
        return self._md_thresh

    @property
    def std_factor(self):
        """Returns the k value calculated using Chebyshev's inequality."""
        return self._k

    @staticmethod
    def _chebyshev_k(p: float) -> float:
        """Calculate the k value using Chebyshev's inequality."""
        return np.reciprocal(np.sqrt(p))

    def fit(self, x: npt.NDArray[float]) -> Self:
        """
        Fit the estimator on the training set.

        Args:
        ----
            x: training data

        Returns
        -------
            self
        """
        self._distr_mean = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False)
        self._cov_inv = np.linalg.inv(cov)
        mahal_dist = self.mahalanobis(x)
        self._md_thresh = np.mean(mahal_dist) + self._k * np.std(mahal_dist)
        self._is_fitted = True
        return self

    def mahalanobis(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Calculate the Mahalanobis distance.

        Args:
        ----
            x: input data

        Returns
        -------
            Mahalanobis distance vector
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
            x: input data

        Returns
        -------
            Integer Array of 0s and 1s

        Raises
        ------
            RuntimeError: if the model is not fitted yet
        """
        if not self._is_fitted:
            raise ModelInitializationError("Model not fitted yet.")
        md = self.mahalanobis(x)
        y_hat = np.zeros_like(x)
        y_hat[md < self._md_thresh] = _INLIER
        y_hat[md >= self._md_thresh] = _OUTLIER
        return y_hat

    def score_samples(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Returns the outlier score for each sample.

        Score is calculated as the ratio of Mahalanobis distance to threshold.
        A score greater than 1.0 means that the sample is an outlier.

        Args:
        ----
            x: input data

        Returns
        -------
            Outlier score for each sample

        Raises
        ------
            RuntimeError: if the model is not fitted yet
        """
        if not self._is_fitted:
            raise ModelInitializationError("Model not fitted yet.")
        return self.mahalanobis(x) / self._md_thresh
