import numpy as np
import numpy.typing as npt
from typing_extensions import Self, Final

from numalogic.base import BaseThresholdModel
from numalogic.tools.exceptions import InvalidDataShapeError, ModelInitializationError

import logging

LOGGER = logging.getLogger(__name__)
_INLIER: Final[int] = 0
_OUTLIER: Final[int] = 1
_INPUT_DIMS: Final[int] = 2


class MaxPercentileThreshold(BaseThresholdModel):
    """
    Percentile based Thresholding estimator.

    Args:
        max_inlier_percentile: Max percentile greater than which will be treated as outlier
        min_threshold:  Value to be used if threshold is less than this
    """

    __slots__ = (
        "_max_percentile",
        "_min_thresh",
        "_thresh",
        "_is_fitted",
        "_adjust_threshold",
        "_adjust_factor",
    )

    def __init__(
        self,
        max_inlier_percentile: float = 96.0,
        min_threshold: float = 1e-4,
        adjust_threshold: bool = False,
        adjust_factor: float = 3.0,
    ):
        super().__init__()
        self._max_percentile = max_inlier_percentile
        self._min_thresh = min_threshold
        self._thresh = None
        self._is_fitted = False
        self._adjust_threshold = adjust_threshold
        self._adjust_factor = adjust_factor

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

        if self._adjust_threshold:
            for idx, _ in enumerate(self._thresh):
                ratio = self._thresh[idx] / self._min_thresh
                if ratio < 1e-2:
                    LOGGER.info(
                        "Min threshold ratio: %s is less than 1e-2 times the "
                        "threshold for column %s;",
                        ratio,
                        idx,
                    )
                    self._thresh[idx] = self._min_thresh * self._adjust_factor

        self._thresh[self._thresh < self._min_thresh] = self._min_thresh
        self._is_fitted = True
        return self

    def predict(self, x: npt.NDArray[float]) -> npt.NDArray[int]:
        if not self._is_fitted:
            raise ModelInitializationError("Model not fitted yet.")
        self._validate_input(x)

        y_hat = np.zeros(x.shape, dtype=int)
        y_hat[x > self._thresh] = _OUTLIER
        return y_hat

    def score_samples(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        if not self._is_fitted:
            raise ModelInitializationError("Model not fitted yet.")

        self._validate_input(x)
        return x / self._thresh
