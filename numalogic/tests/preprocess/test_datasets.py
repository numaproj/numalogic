import os
import unittest

import numpy as np
import pandas as pd

from numalogic._constants import TESTS_DIR
from numalogic.preprocess.datasets import SequenceDataset

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")


class TestSequenceDataset(unittest.TestCase):
    df: pd.DataFrame = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        cls.df = df[["success", "failure"]]

    def test_create_dataset_df(self):
        dataset = SequenceDataset(self.df, 120)
        self.assertTrue(len(dataset))
        self.assertEqual(2, dataset.data.shape[1])
        self.assertEqual(120, dataset.data.shape[2])

    def test_create_dataset_ndarr(self):
        dataset = SequenceDataset(self.df.to_numpy(), 120)
        self.assertTrue(len(dataset))
        self.assertEqual(2, dataset.data.shape[1])
        self.assertEqual(120, dataset.data.shape[2])

    def test_recover_shape(self):
        dataset = SequenceDataset(self.df.to_numpy(), 120)
        recovered = dataset.recover_shape(dataset.data)
        self.assertEqual(recovered.shape, self.df.shape)
        print(np.mean(recovered))
        print(self.df.mean())
        self.assertAlmostEqual(np.mean(recovered), np.mean(self.df.to_numpy()), places=5)


if __name__ == "__main__":
    unittest.main()
