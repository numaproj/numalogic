import unittest

import numpy as np

from numalogic.blocks import Block
from sklearn.ensemble import IsolationForest


class DummyBlock(Block):
    def fit(self, input_: np.ndarray, **__) -> np.ndarray:
        return self._artifact.fit_predict(input_).reshape(-1, 1)

    def run(self, input_: np.ndarray, **__) -> np.ndarray:
        return self._artifact.predict(input_).reshape(-1, 1)


class TestBlock(unittest.TestCase):
    def test_random_block(self):
        block = DummyBlock(IsolationForest(), name="isolation_forest")
        self.assertEqual(block.name, "isolation_forest")

        block.fit(np.arange(100).reshape(-1, 2))
        out = block(np.arange(10).reshape(-1, 2))
        self.assertEqual(out.shape, (5, 1))

        self.assertIsInstance(block.artifact, IsolationForest)
        self.assertIsInstance(block.artifact_state, IsolationForest)
        self.assertTrue(block.stateful)


if __name__ == "__main__":
    unittest.main()
