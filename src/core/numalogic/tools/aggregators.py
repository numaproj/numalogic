from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from numalogic.transforms import expmov_avg_aggregator


def aggregate_window(
    y: npt.NDArray[float], agg_func: Callable = expmov_avg_aggregator, **agg_func_kw
) -> npt.NDArray[float]:
    """Aggregate over window/sequence length."""
    return np.apply_along_axis(func1d=agg_func, axis=0, arr=y, **agg_func_kw).reshape(-1)


def aggregate_features(
    y: npt.NDArray[float], agg_func: Callable = np.mean, **agg_func_kw
) -> npt.NDArray[float]:
    """Aggregate over features."""
    return agg_func(y, axis=1, keepdims=True, **agg_func_kw)
