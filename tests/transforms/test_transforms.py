import warnings

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_less,
    assert_array_equal,
    assert_allclose,
)
from sklearn.pipeline import make_pipeline

from numalogic.base import StatelessTransformer
from numalogic.transforms import (
    LogTransformer,
    StaticPowerTransformer,
    TanhScaler,
    DataClipper,
    GaussianNoiseAdder,
    DifferenceTransform,
    FlattenVector,
    PercentileScaler,
    FlattenVectorWithPadding,
)

RNG = np.random.default_rng(42)


def test_logtransformer():
    x = 3 + RNG.random((5, 3))
    transformer = LogTransformer(add_factor=1)
    x_prime = transformer.transform(x)

    assert_almost_equal(np.log1p(x), x_prime)
    assert_almost_equal(transformer.fit_transform(x), x_prime)
    assert_almost_equal(transformer.inverse_transform(x_prime), np.expm1(x_prime))


def test_staticpowertransformer():
    x = 3 + RNG.random((5, 3))
    transformer = StaticPowerTransformer(3, add_factor=4)
    x_prime = transformer.transform(x)

    assert_almost_equal(np.power(4 + x, 3), x_prime)
    assert_almost_equal(transformer.fit_transform(x), x_prime)
    assert_almost_equal(transformer.inverse_transform(x_prime), x, decimal=3)


def test_tanh_scaler_1():
    x = 1 + RNG.random((5, 3))
    scaler = TanhScaler()
    x_scaled = scaler.fit_transform(x)

    assert_array_less(x_scaled, np.ones_like(x_scaled))
    assert_array_less(np.zeros_like(x_scaled), x_scaled)


def test_tanh_scaler_2():
    x = 3 + RNG.random((5, 3))
    pl = make_pipeline(LogTransformer(), TanhScaler())

    x_scaled = pl.fit_transform(x)
    assert_array_less(x_scaled, np.ones_like(x_scaled))
    assert_array_less(np.zeros_like(x_scaled), x_scaled)


def test_tanh_scaler_3():
    x = RNG.random((5, 3))
    x[:, 1] = np.zeros(5)

    scaler = TanhScaler()

    x_scaled = scaler.fit_transform(x)
    assert not np.isnan(x_scaled[:, 1]).all()
    assert_array_less(x_scaled, np.ones_like(x_scaled))
    assert_array_less(np.zeros_like(x_scaled), x_scaled)


def test_tanh_scaler_nan():
    x = RNG.random((5, 3))
    x[:, 1] = np.zeros(5)

    scaler = TanhScaler(eps=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_scaled = scaler.fit_transform(x)
    assert np.isnan(x_scaled[:, 1]).all()


def test_base_transform():
    x = RNG.random((5, 3))
    x[:, 1] = np.zeros(5)

    trfr = StatelessTransformer()
    with pytest.raises(NotImplementedError):
        trfr.transform(x)
        trfr.fit_transform(x)
    assert trfr.fit(x) == trfr


def test_gaussian_noise_adder():
    x = np.zeros((5, 3))

    tx = GaussianNoiseAdder()
    x_ = tx.transform(x)

    assert x.shape == x_.shape
    assert_almost_equal(
        np.mean(x),
        np.mean(x_),
    )


def test_dataclipper():
    x = np.ones((5, 2))

    tx1 = DataClipper(upper=0.1)
    x_ = tx1.transform(x)

    assert x.shape == x_.shape
    assert_array_equal(np.asarray([0.1, 0.1], dtype=np.float32), np.max(x_, axis=0))


def test_dataclipper_1():
    x = np.ones((5, 2))

    tx1 = DataClipper(upper=[0.8, 0.9])
    tx2 = DataClipper(lower=[0.1, 0.2])
    x_ = tx1.transform(x)
    x_ = tx2.transform(x_)
    assert x.shape == x_.shape
    assert_array_equal(np.asarray([0.8, 0.9], dtype=np.float32), np.max(x_, axis=0))


def test_dataclipper_2():
    x = np.ones((3, 3))
    x[:, 1] = np.zeros(3)

    tx = DataClipper(lower=[1.0, "-inf", 0.0], upper=[1.0, "inf", 0.7])
    x_ = tx.transform(x)
    assert x.shape == x_.shape
    assert_array_equal(np.asarray([1.0, 0, 0.7], dtype=np.float32), np.max(x_, axis=0))


def test_dataclipper_3():
    np.ones((5, 3))
    with pytest.raises(ValueError):
        DataClipper(upper=[0.8, 0.1, 0.1], lower=[0.8, 0.2, 0.2])
    with pytest.raises(ValueError):
        DataClipper(upper=[0.8, 0.1, 0.1], lower=[0.8, 0.2, 0.2])
    with pytest.raises(ValueError):
        DataClipper()


def test_dataclipper_4():
    np.ones((5, 2))
    with pytest.raises(ValueError):
        DataClipper(upper=[0.8, 0.9, 0.1], lower=2.0)


def test_difftx():
    x = np.ones((5, 3))
    x[1, :] = 0
    x[4, :] = 2

    tx = DifferenceTransform()
    x_ = tx.transform(x)

    assert x.shape == x_.shape
    assert_array_equal(
        x_,
        np.array([[-1, -1, -1], [-1, -1, -1], [1, 1, 1], [0, 0, 0], [1, 1, 1]]),
    )


def test_flattenvector():
    x = RNG.random((5, 2))
    clf = FlattenVector(n_features=2)
    data = clf.transform(x)

    assert data.shape[1] == 1
    assert clf.inverse_transform(data).shape[1] == 2


def test_flattenvector_with_padding():
    x = RNG.random((10, 4))
    features = ["failed", "degraded", "errorrate", "userimpact"]
    flatten_features = ["failed", "degraded"]

    clf = FlattenVectorWithPadding(features, flatten_features)
    X_transformed = clf.transform(x)
    assert X_transformed.shape[1] == (len(features) - len(flatten_features) + 1)

    X_inverse = clf.inverse_transform(X_transformed)
    assert X_inverse.shape == x.shape


def test_percentile_scaler():
    x = np.abs(RNG.random((10, 3)))
    tx = PercentileScaler(max_percentile=99, eps=1e-6)
    x_ = tx.fit_transform(x)
    assert x.shape == x_.shape
    assert_allclose(tx.data_pth_max, np.percentile(x, 99, axis=0))
    assert_allclose(tx.data_pth_min, np.min(x, axis=0))
