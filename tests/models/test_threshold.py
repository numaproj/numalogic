import unittest

import numpy as np

from numalogic.models.threshold import StdDevThreshold, StaticThreshold


class TestStdDevThreshold(unittest.TestCase):
    def setUp(self) -> None:
        self.x_train = np.arange(100).reshape(50, 2)
        self.x_test = np.arange(100, 160, 6).reshape(5, 2)

    def test_estimator_predict(self):
        clf = StdDevThreshold()
        clf.fit(self.x_train)
        y = clf.predict(self.x_test)
        self.assertAlmostEqual(0.4, np.mean(y), places=1)

    def test_estimator_score(self):
        clf = StdDevThreshold()
        clf.fit(self.x_train)
        score = clf.score_samples(self.x_test)
        self.assertAlmostEqual(0.93317, np.mean(score), places=2)


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


if __name__ == "__main__":
    unittest.main()
