import pytest
import numpy as np

from numalogic.models.threshold import StaticThreshold, SigmoidThreshold


@pytest.fixture
def x():
    return np.arange(20).reshape(10, 2).astype(float)


def test_static_threshold_predict(x):
    clf = StaticThreshold(upper_limit=5)
    clf.fit(x)
    y = clf.predict(x)
    assert x.shape == y.shape
    assert np.max(y) == 1
    assert np.min(y) == 0


def test_static_threshold_score_01(x):
    clf = StaticThreshold(upper_limit=5.0)
    y = clf.score_samples(x)
    assert x.shape == y.shape
    assert np.max(y) == clf.outlier_score
    assert np.min(y) == clf.inlier_score


def test_sigmoid_threshold_score_02(x):
    clf = SigmoidThreshold(5, 10)
    y = clf.score_samples(x)
    assert x.shape == y.shape
    assert np.max(y) == clf.score_limit
    assert np.min(y) > 0.0


def test_sigmoid_threshold_predict(x):
    clf = SigmoidThreshold(5)
    clf.fit(x)
    y = clf.predict(x)
    assert x.shape == y.shape
    assert np.max(y) == 1
    assert np.min(y) == 0
