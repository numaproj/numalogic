import numpy as np
import pytest

from numalogic.models.threshold import MahalanobisThreshold, RobustMahalanobisThreshold
from numalogic.tools.exceptions import ModelInitializationError, InvalidDataShapeError


@pytest.fixture
def rng_data():
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(100, 15))
    x_test = rng.normal(size=(30, 15))
    return x_train, x_test


class TestMahalanobisThreshold:
    def test_init(self, rng_data):
        x_train, x_test = rng_data
        clf = MahalanobisThreshold(max_outlier_prob=0.25)
        clf.fit(x_train)
        md = clf.mahalanobis(x_test)
        assert (x_test.shape[0],) == md.shape
        assert all(md) > 0.0
        assert clf.threshold > 0.0
        assert clf.std_factor == 2.0

    def test_init_err(self):
        with pytest.raises(ValueError):
            MahalanobisThreshold(max_outlier_prob=0.0)
        with pytest.raises(ValueError):
            MahalanobisThreshold(max_outlier_prob=1.0)

    def test_singular(self):
        clf = MahalanobisThreshold()
        clf.fit(np.ones((100, 15)))
        md = clf.mahalanobis(np.ones((30, 15)))
        assert (30,) == md.shape

    def test_predict(self, rng_data):
        x_train, x_test = rng_data
        clf = MahalanobisThreshold()
        clf.fit(x_train)
        y = clf.predict(x_test)
        assert (x_test.shape[0],) == y.shape
        assert np.max(y) == 1
        assert np.min(y) == 0

    def test_notfitted_err(self, rng_data):
        x_test = rng_data[1]
        clf = MahalanobisThreshold()
        with pytest.raises(ModelInitializationError):
            clf.predict(x_test)
        with pytest.raises(ModelInitializationError):
            clf.score_samples(x_test)

    def test_invalid_input_err(self, rng_data):
        x_train = rng_data[0]
        clf = MahalanobisThreshold()
        clf.fit(x_train)
        with pytest.raises(InvalidDataShapeError):
            clf.predict(np.ones((30, 15, 1)))
        with pytest.raises(InvalidDataShapeError):
            clf.score_samples(np.ones(30))

    def test_score_samples(self, rng_data):
        x_train, x_test = rng_data
        clf = MahalanobisThreshold()
        clf.fit(x_train)
        y = clf.score_samples(x_test)
        assert (x_test.shape[0],) == y.shape

    def test_score_samples_err(self, rng_data):
        x_test = rng_data[1]
        clf = MahalanobisThreshold()
        with pytest.raises(ModelInitializationError):
            clf.score_samples(x_test)


class TestRobustMahalanobisThreshold:
    def test_fit(self, rng_data):
        x_train, x_test = rng_data
        clf = RobustMahalanobisThreshold(max_outlier_prob=0.25)
        clf.fit(x_train)
        md = clf.mahalanobis(x_test)
        assert (x_test.shape[0],) == md.shape
        assert all(md) > 0.0
        assert clf.threshold > 0.0
        assert clf.std_factor == 2.0

    def test_score(self, rng_data):
        x_train, x_test = rng_data
        clf = MahalanobisThreshold()
        clf.fit(x_train)
        y = clf.score_samples(x_test)
        assert (x_test.shape[0],) == y.shape
