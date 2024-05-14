import numpy as np
from sklearn.pipeline import make_pipeline
import pytest

from numalogic.transforms import (
    tanh_norm,
    TanhNorm,
    ExpMovingAverage,
    SigmoidNorm,
    expmov_avg_aggregator,
)


def test_tanh_norm_func():
    arr = np.arange(10)
    scores = tanh_norm(arr)
    assert sum(scores) == pytest.approx(39.52, 0.01)  # places=2


def test_tanh_norm_clf():
    arr = np.arange(10).reshape(5, 2)
    clf = TanhNorm()
    scores = clf.fit_transform(arr)

    assert arr.shape == scores.shape
    assert np.sum(scores) == pytest.approx(39.52, 0.01)  # places=2


def test_exp_mov_avg_estimator():
    beta = 0.9
    arr = np.arange(1, 11).reshape(-1, 1)
    clf = ExpMovingAverage(beta)
    out = clf.fit_transform(arr)

    expected = expmov_avg_aggregator(arr, beta)

    assert arr.shape == out.shape
    assert expected == pytest.approx(out[-1].item(), 0.01)  # places=2
    assert out.data.c_contiguous


def test_exp_mov_avg_estimator_err():
    with pytest.raises(ValueError):
        ExpMovingAverage(1.1)

    with pytest.raises(ValueError):
        ExpMovingAverage(0.0)

    with pytest.raises(ValueError):
        ExpMovingAverage(1.0)


def test_exp_mov_avg_agg():
    arr = np.arange(1, 11)
    val = expmov_avg_aggregator(arr, 0.9)
    assert isinstance(val, float)
    assert val < 10


def test_exp_mov_avg_agg_err():
    arr = np.arange(1, 11)
    with pytest.raises(ValueError):
        expmov_avg_aggregator(arr, 1.01)


def test_postproc_pl():
    x = np.arange(1, 11).reshape(-1, 1)
    pl = make_pipeline(TanhNorm(), ExpMovingAverage(0.9))
    out = pl.transform(x)
    assert x.shape == out.shape


def test_sig_norm():
    x = np.arange(1, 11).reshape(-1, 1)
    clf = SigmoidNorm()
    out = clf.fit_transform(x)
    assert x.shape == out.shape
    assert out.data.c_contiguous
    assert np.all(out >= 0)
    assert np.all(out <= 10)
