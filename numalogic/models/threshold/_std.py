import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator


class StdDevThreshold(BaseEstimator):
    r"""
    Threshold estimator that calculates based on the mean and the std deviation.

    Threshold = Mean + (std_factor * Std)

    Generates anomaly score as the ratio
    between the input data and threshold generated.
    """

    def __init__(self, std_factor: float = 3.0, min_threshold: float = 0.1):
        self.std_factor = std_factor
        self.min_threshold = min_threshold

        self._std = None
        self._mean = None
        self._threshold = None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def threshold(self):
        return self._threshold

    def fit(self, x_train: NDArray[float], y=None) -> "StdDevThreshold":
        self._std = np.std(x_train, axis=0)
        self._mean = np.mean(x_train, axis=0)
        self._threshold = self._mean + (self.std_factor * self._std)
        self._threshold[self._threshold < self.min_threshold] = self.min_threshold

        return self

    def predict(self, x_test: NDArray[float]) -> NDArray[float]:
        anomaly_scores = x_test / self.threshold
        return anomaly_scores

    def score(self, x_test: NDArray[float]) -> NDArray[float]:
        return self.predict(x_test)
