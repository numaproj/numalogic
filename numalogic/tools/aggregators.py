import os
from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt

from numalogic.transforms import expmov_avg_aggregator


EXP_MOV_AVG_BETA = float(os.getenv("EXP_MOV_AVG_BETA", "0.6"))


def aggregate_window(y: npt.NDArray[float]) -> npt.NDArray[float]:
    """Aggregate over window/sequence length."""
    return np.apply_along_axis(
        func1d=expmov_avg_aggregator, axis=0, arr=y, beta=EXP_MOV_AVG_BETA
    ).reshape(-1)


def aggregate_features(
    y: npt.NDArray[float], weights: Optional[Sequence[float]] = None
) -> npt.NDArray[float]:
    """Aggregate over features."""
    if weights:
        return np.average(y, weights=weights, axis=1, keepdims=True)
    return np.mean(y, axis=1, keepdims=True)
