import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator


class StdDevThreshold(BaseEstimator):
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

    def fit(self, X, y=None):
        self._std = np.std(X, axis=0)
        self._mean = np.mean(X, axis=0)
        self._threshold = self._mean + (self.std_factor * self._std)
        self._threshold[self._threshold < self.min_threshold] = self.min_threshold

        return self

    def predict(self, X: NDArray[float]) -> NDArray[float]:
        anomaly_scores = X / self.threshold
        return anomaly_scores
