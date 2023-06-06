import unittest

import numpy as np

from numalogic.blocks import Block
from sklearn.ensemble import IsolationForest


class RandomBlock(Block):
    pass


class TestBlock(unittest.TestCase):
    def test_random_block(self):
        block = RandomBlock(IsolationForest(), name="isolation_forest")
        self.assertEqual(block.name, "isolation_forest")
        self.assertRaises(NotImplementedError, block.fit, np.arange(100).reshape(-1, 2))
        self.assertRaises(NotImplementedError, block.run, np.arange(10).reshape(-1, 2))
        self.assertIsInstance(block.artifact, IsolationForest)
        self.assertIsInstance(block.artifact_state, IsolationForest)
        self.assertTrue(block.stateful)


if __name__ == "__main__":
    unittest.main()
