import os

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from numalogic._constants import TESTS_DIR
from numalogic.models.threshold import MaxPercentileThreshold


@pytest.fixture
def data() -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    x = pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "prom_mv.csv"), index_col="timestamp"
    ).to_numpy(dtype=np.float32)
    return x[:-50], x[-50:]


@pytest.fixture()
def fitted(data):
    clf = MaxPercentileThreshold(max_inlier_percentile=75, min_threshold=1e-3)
    x_train, _ = data
    clf.fit(x_train)
    return clf


def test_score_samples(data, fitted):
    _, x_test = data
    y_scores = fitted.score_samples(x_test)
    assert len(fitted.threshold) == 3
    assert fitted.threshold[1] == 1e-3
    assert y_scores.shape == (50, 3)


def test_predict(data, fitted):
    _, x_test = data
    y_pred = fitted.predict(x_test)
    assert y_pred.shape == (50, 3)
