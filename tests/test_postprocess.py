import unittest

import numpy as np

from numalogic.postprocess import tanh_norm, TanhNorm


class TestPostprocess(unittest.TestCase):
    def test_tanh_norm_func(self):
        arr = np.arange(10)
        scores = tanh_norm(arr)
        print(scores)

        self.assertAlmostEqual(sum(scores), 39.52, places=2)

    def test_tanh_norm_clf(self):
        arr = np.arange(10).reshape(5, 2)
        clf = TanhNorm()
        scores = clf.fit_transform(arr)

        self.assertTupleEqual(arr.shape, scores.shape)
        self.assertAlmostEqual(np.sum(scores), 39.52, places=2)


if __name__ == "__main__":
    unittest.main()
