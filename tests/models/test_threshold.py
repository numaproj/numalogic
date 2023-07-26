import unittest

import numpy as np

from numalogic.models.threshold import (
    StdDevThreshold,
    StaticThreshold,
    SigmoidThreshold,
    MahalanobisThreshold,
)
from numalogic.tools.exceptions import ModelInitializationError, InvalidDataShapeError


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


class TestMahalanobisThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)
        cls.x_train = rng.normal(size=(100, 15))
        cls.x_test = rng.normal(size=(30, 15))

    def test_init(self):
        clf = MahalanobisThreshold(max_outlier_prob=0.25)
        clf.fit(self.x_train)
        md = clf.mahalanobis(self.x_test)
        self.assertTupleEqual((self.x_test.shape[0],), md.shape)
        self.assertGreater(all(md), 0.0)
        self.assertGreater(clf.threshold, 0.0)
        self.assertEqual(clf.std_factor, 2.0)

    def test_init_err(self):
        with self.assertRaises(ValueError):
            MahalanobisThreshold(max_outlier_prob=0.0)
        with self.assertRaises(ValueError):
            MahalanobisThreshold(max_outlier_prob=1.0)

    def test_singular(self):
        clf = MahalanobisThreshold()
        clf.fit(np.ones((100, 15)))
        md = clf.mahalanobis(np.ones((30, 15)))
        self.assertTupleEqual((30,), md.shape)

    def test_predict(self):
        clf = MahalanobisThreshold()
        clf.fit(self.x_train)
        y = clf.predict(self.x_test)
        self.assertTupleEqual((self.x_test.shape[0],), y.shape)
        self.assertEqual(np.max(y), 1)
        self.assertEqual(np.min(y), 0)

    def test_notfitted_err(self):
        clf = MahalanobisThreshold()
        with self.assertRaises(ModelInitializationError):
            clf.predict(self.x_test)
        with self.assertRaises(ModelInitializationError):
            clf.score_samples(self.x_test)

    def test_invalid_input_err(self):
        clf = MahalanobisThreshold()
        clf.fit(self.x_train)
        with self.assertRaises(InvalidDataShapeError):
            clf.predict(np.ones((30, 15, 1)))
        with self.assertRaises(InvalidDataShapeError):
            clf.score_samples(np.ones(30))

    def test_score_samples(self):
        clf = MahalanobisThreshold()
        clf.fit(self.x_train)
        y = clf.score_samples(self.x_test)
        self.assertTupleEqual((self.x_test.shape[0],), y.shape)

    def test_score_samples_err(self):
        clf = MahalanobisThreshold()
        with self.assertRaises(ModelInitializationError):
            clf.score_samples(self.x_test)


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
