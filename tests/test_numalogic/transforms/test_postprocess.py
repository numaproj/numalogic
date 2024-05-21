import unittest

import numpy as np
from sklearn.pipeline import make_pipeline

from numalogic.transforms import tanh_norm, TanhNorm, ExpMovingAverage, expmov_avg_aggregator
from numalogic.tools.exceptions import InvalidDataShapeError


class TestPostprocess(unittest.TestCase):
    def test_tanh_norm_func(self):
        arr = np.arange(10)
        scores = tanh_norm(arr)

        self.assertAlmostEqual(sum(scores), 39.52, places=2)

    def test_tanh_norm_clf(self):
        arr = np.arange(10).reshape(5, 2)
        clf = TanhNorm()
        scores = clf.fit_transform(arr)

        self.assertTupleEqual(arr.shape, scores.shape)
        self.assertAlmostEqual(np.sum(scores), 39.52, places=2)

    def test_exp_mov_avg_estimator_01(self):
        arr = np.arange(1, 11).reshape(-1, 1)
        clf = ExpMovingAverage(0.9)
        out = clf.fit_transform(arr)
        self.assertTupleEqual(arr.shape, out.shape)
        self.assertAlmostEqual(expmov_avg_aggregator(arr, 0.9), out[-1].item(), places=3)

    def test_exp_mov_avg_estimator_02(self):
        arr = np.arange(1, 11).reshape(-1, 1)
        clf = ExpMovingAverage(0.9, bias_correction=False)
        out = clf.fit_transform(arr)
        self.assertTupleEqual(arr.shape, out.shape)
        self.assertAlmostEqual(
            expmov_avg_aggregator(arr, 0.9, bias_correction=False), out[-1].item(), places=3
        )

    def test_exp_mov_avg_estimator_err(self):
        with self.assertRaises(ValueError):
            ExpMovingAverage(1.1)

        with self.assertRaises(ValueError):
            ExpMovingAverage(0.0)

        with self.assertRaises(ValueError):
            ExpMovingAverage(1.0)

        with self.assertRaises(InvalidDataShapeError):
            clf = ExpMovingAverage(0.1)
            clf.fit_transform(np.arange(10).reshape(1, -1))

        with self.assertRaises(InvalidDataShapeError):
            clf = ExpMovingAverage(0.1)
            clf.fit_transform(np.arange(12).reshape((2, 3, -1)))

    def test_exp_mov_avg_agg(self):
        arr = np.arange(1, 11)
        val = expmov_avg_aggregator(arr, 0.9)
        self.assertIsInstance(val, float)
        self.assertLess(val, 10)

    def test_exp_mov_avg_agg_err(self):
        arr = np.arange(1, 11)
        with self.assertRaises(ValueError):
            expmov_avg_aggregator(arr, 1.01)

    def test_postproc_pl(self):
        x = np.arange(1, 11).reshape(-1, 1)
        pl = make_pipeline(TanhNorm(), ExpMovingAverage(0.9))
        out = pl.transform(x)
        self.assertTupleEqual(x.shape, out.shape)


if __name__ == "__main__":
    unittest.main()
