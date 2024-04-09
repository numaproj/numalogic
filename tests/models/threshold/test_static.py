import logging

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from numalogic.models.threshold import StaticThreshold, SigmoidThreshold


logging.basicConfig(level=logging.DEBUG)


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
    clf = SigmoidThreshold(2.0, 1e100)
    y = clf.score_samples(x)
    assert x.shape == y.shape
    assert_almost_equal(np.max(y, axis=0)[0], clf.score_limit)
    assert_almost_equal(np.max(y, axis=0)[1], 0.0)
    assert y.dtype == np.float32


def test_sigmoid_threshold_predict(x):
    clf = SigmoidThreshold(5)
    clf.fit(x)
    y = clf.predict(x)

    assert x.shape == y.shape
    assert np.max(y) == 1
    assert np.min(y) == 0
