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


import numpy.typing as npt
from sklearn.base import BaseEstimator
from typing_extensions import Self


class StaticThreshold(BaseEstimator):
    r"""
    Simple and stateless static thresholding as an estimator.

    Values more than upper_limit is considered an outlier,
    and are given an outlier_score.

    Values less than the upper_limit is considered an inlier,
    and are given an inlier_score.

    Args:
        upper_limit: upper threshold
        outlier_score: static score given to values above upper threshold;
                       this has to be greater than inlier_score
        inlier_score: static score given to values below upper threshold
    """
    __slots__ = ("upper_limit", "outlier_score", "inlier_score")

    def __init__(self, upper_limit: float, outlier_score: float = 10.0, inlier_score: float = 0.5):
        self.upper_limit = float(upper_limit)
        self.outlier_score = float(outlier_score)
        self.inlier_score = float(inlier_score)

        assert (
            self.outlier_score > self.inlier_score
        ), "Outlier score needs to be greater than inlier score"

    def fit(self, _: npt.NDArray[float]) -> Self:
        """Does not do anything. Only for API compatibility"""
        return self

    def predict(self, x_test: npt.NDArray[float]) -> npt.NDArray[int]:
        """
        Returns an integer array of same shape as input.
        1 denotes anomaly.
        """
        y_test = x_test.copy()
        y_test[x_test < self.upper_limit] = 0
        y_test[x_test >= self.upper_limit] = 1
        return y_test

    def score_samples(self, x_test: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Returns an array of same shape as input
        with values being anomaly scores.
        """
        x_test = x_test.copy()
        x_test[x_test < self.upper_limit] = self.inlier_score
        x_test[x_test >= self.upper_limit] = self.outlier_score
        return x_test
