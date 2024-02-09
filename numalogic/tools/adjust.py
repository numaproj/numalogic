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

from numalogic.models.threshold import SigmoidThreshold


class ScoreAdjuster:
    """
    Adjusts the model output based on the metric input.

    Args:
    ----
        adjust_weight: weight given to static thresholding output
            (between 0 and 1)
        metric_weights: weights given to each kpi/metric
        upper_limits: upper limits for each metric
        kwargs: kwargs for SigmoidThreshold

    Raises
    ------
        ValueError: if adjust_weight is not between 0 and 1
    """

    __slots__ = ("_adjust_wt", "_kpi_wts", "_thresholder")

    def __init__(
        self, adjust_weight: float, metric_weights: list[float], upper_limits: list[float], **kwargs
    ):
        if adjust_weight <= 0.0 or adjust_weight >= 1:
            raise ValueError("adjust_weight needs to be between 0 and 1")
        self._adjust_wt = adjust_weight
        self._kpi_wts = np.asarray(metric_weights, dtype=np.float32).reshape(-1, 1)
        self._thresholder = SigmoidThreshold(*upper_limits, **kwargs)

    def adjust(
        self, metric_in: npt.NDArray[float], model_scores: npt.NDArray[float]
    ) -> npt.NDArray[float]:
        """
        Adjusts the model output based on the metric input.

        Args:
        ----
            metric_in: metric input to the model
            model_scores: model output scores

        Returns
        -------
            adjusted_scores: adjusted scores
        """
        model_scores = np.reshape(-1, 1)
        feature_scores = self._thresholder.score_samples(metric_in)
        weighted_scores = np.dot(feature_scores, self._kpi_wts)
        return (self._adjust_wt * weighted_scores) + ((1 - self._adjust_wt) * model_scores)

    # @classmethod
    # def from_conf(cls, conf: ScoreAdjustConf) -> Self:
    #     """
    #     Creates an instance of ScoreAdjuster from ScoreAdjustConf.
    #
    #     Args:
    #     ----
    #         conf: ScoreAdjustConf
    #
    #     Returns
    #     -------
    #         ScoreAdjuster instance
    #     """
    #     return cls(conf.weight, conf.metric_weights, conf.upper_limits)
