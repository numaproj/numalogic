import unittest

import numpy as np

from numalogic.models.threshold import StdDevThreshold


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


if __name__ == "__main__":
    unittest.main()
