import unittest

import numpy as np

from numalogic.scores import tanh_norm


class TestScores(unittest.TestCase):
    def test_tanh_norm(self):
        arr = np.arange(10)
        scores = tanh_norm(arr)
        print(scores)

        self.assertAlmostEqual(sum(scores), 39.52, places=2)


if __name__ == "__main__":
    unittest.main()
