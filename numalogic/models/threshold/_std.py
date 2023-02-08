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
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from typing_extensions import Self


class StdDevThreshold(BaseEstimator):
    r"""
    Threshold estimator that calculates based on the mean and the std deviation.

    Threshold = Mean + (std_factor * Std)

    Generates anomaly score as the ratio
    between the input data and threshold generated.

    Args:
        std_factor: scaler factor for std to be added to mean
        min_threshold: clip the threshold value to be above this value
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

    def fit(self, x_train: NDArray[float], y=None) -> Self:
        """
        Fit the estimator on the training set.
        """
        self._std = np.std(x_train, axis=0)
        self._mean = np.mean(x_train, axis=0)
        self._threshold = self._mean + (self.std_factor * self._std)
        self._threshold[self._threshold < self.min_threshold] = self.min_threshold

        return self

    def predict(self, x_test: NDArray[float]) -> NDArray[int]:
        """
        Returns an integer array of same shape as input.
        1 denotes outlier, 0 denotes inlier
        """
        y_pred = x_test.copy()
        y_pred[x_test < self._threshold] = 0
        y_pred[x_test >= self._threshold] = 1
        return y_pred

    def score_samples(self, x_test: NDArray[float]) -> NDArray[float]:
        """
        Returns an anomaly score array with the same shape as input.
        """
        y_scores = x_test / self.threshold
        return y_scores
