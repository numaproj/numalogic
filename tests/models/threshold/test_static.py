import unittest

import numpy as np

from numalogic.models.threshold import StaticThreshold, SigmoidThreshold


class TestStaticThreshold(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.arange(20).reshape(10, 2).astype(float)

    def test_predict(self):
        clf = StaticThreshold(upper_limit=5)
        clf.fit(self.x)
        y = clf.predict(self.x)
        self.assertTupleEqual(self.x.shape, y.shape)
        self.assertEqual(np.max(y), 1)
        self.assertEqual(np.min(y), 0)

    def test_score(self):
        clf = StaticThreshold(upper_limit=5.0)
        y = clf.score_samples(self.x)
        self.assertTupleEqual(self.x.shape, y.shape)
        self.assertEqual(np.max(y), clf.outlier_score)
        self.assertEqual(np.min(y), clf.inlier_score)


class TestSigmoidThreshold(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.arange(20).reshape(10, 2).astype(float)

    def test_predict(self):
        clf = SigmoidThreshold(upper_limit=5)
        clf.fit(self.x)
        y = clf.predict(self.x)
        self.assertTupleEqual(self.x.shape, y.shape)
        self.assertEqual(np.max(y), 1)
        self.assertEqual(np.min(y), 0)

    def test_score(self):
        clf = SigmoidThreshold(upper_limit=5.0)
        y = clf.score_samples(self.x)
        self.assertTupleEqual(self.x.shape, y.shape)
        self.assertEqual(np.max(y), clf.score_limit)
        self.assertGreater(np.min(y), 0.0)


if __name__ == "__main__":
    unittest.main()
