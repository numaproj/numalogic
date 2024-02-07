import numpy as np
import numpy.typing as npt
from typing_extensions import Self, Final

from numalogic.base import BaseThresholdModel
from numalogic.tools.exceptions import InvalidDataShapeError, ModelInitializationError

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

    __slots__ = ("_max_percentile", "_min_thresh", "_thresh", "_is_fitted")

    def __init__(
        self,
        max_inlier_percentile: float = 96.0,
        min_threshold: float = 1e-4,
    ):
        super().__init__()
        self._max_percentile = max_inlier_percentile
        self._min_thresh = min_threshold
        self._thresh = None
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
