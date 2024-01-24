from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt
from typing import Self, Final

from numalogic.base import BaseThresholdModel
from numalogic.tools.exceptions import InvalidDataShapeError, ModelInitializationError

_INLIER: Final[int] = 0
_OUTLIER: Final[int] = 1
_INPUT_DIMS: Final[int] = 2


class MaxPercentileThreshold(BaseThresholdModel):
    def __init__(
        self,
        max_inlier_percentile: float = 96.0,
        min_threshold: float = 1e-3,
        aggregate: bool = False,
        feature_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self._max_percentile = max_inlier_percentile
        self._min_thresh = min_threshold
        self._thresh = None
        self._agg = aggregate
        self._weights = feature_weights
        self._is_fitted = False

    @property
    def threshold(self):
        return self._thresh

    @staticmethod
    def _validate_input(x: npt.NDArray[float]) -> None:
        """Validate the input matrix shape."""
        if x.ndim != _INPUT_DIMS:
            raise InvalidDataShapeError(f"Input matrix should have 2 dims, given shape: {x.shape}.")

    def fit(self, x: npt.NDArray[float]) -> Self:
        self._validate_input(x)
        self._thresh = np.percentile(x, self._max_percentile, axis=0)
        self._thresh[self._thresh < self._min_thresh] = self._min_thresh
        self._is_fitted = True
        return self

    def predict(self):
        pass

    def score_samples(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        if not self._is_fitted:
            raise ModelInitializationError("Model not fitted yet.")

        self._validate_input(x)
        scores = x / self._thresh

        if self._agg:
            return self.agg_score_samples(scores, weights=self._weights)
        return scores

    @staticmethod
    def agg_score_samples(
        y: npt.NDArray[float], weights: Optional[Sequence[float]] = None
    ) -> npt.NDArray[float]:
        if weights:
            return np.average(y, weights=weights, axis=1)
        return np.mean(y, axis=1)
