import copy
import unittest

from numalogic.synthetic import SyntheticTSGenerator, SparsityGenerator


class Testsparsity(unittest.TestCase):
    def test_sparsity_generator1(self):
        ts_generator = SyntheticTSGenerator(12000, 10)
        ts_df = ts_generator.gen_tseries()
        data = copy.deepcopy(ts_df)
        SparsityGN = SparsityGenerator(data, sparse_ratio=0)
        SparsityGN.generate_sparsity()
        transformed_data = SparsityGN.data
        self.assertEqual(transformed_data.equals(ts_df), True)

    def test_sparsity_generator2(self):
        ts_generator = SyntheticTSGenerator(12000, 10)
        ts_df = ts_generator.gen_tseries()
        data = copy.deepcopy(ts_df)
        SparsityGN = SparsityGenerator(data, sparse_ratio=1)
        SparsityGN.generate_sparsity()
        transformed_data = SparsityGN.data
        self.assertEqual(transformed_data.equals(ts_df), False)

    def test_sparsityreturn_series(self):
        ts_generator = SyntheticTSGenerator(12000, 10)
        ts_df = ts_generator.gen_tseries()
        data = copy.deepcopy(ts_df)
        SparsityGN = SparsityGenerator(data, sparse_ratio=0)
        SparsityGN.generate_sparsity()
        transformed_data = SparsityGN.data
        self.assertEqual(transformed_data.shape, ts_df.shape)


if __name__ == "__main__":
    unittest.main()
