import unittest

from numalogic.synthetic import SyntheticTSGenerator


class TestSyntheticTSGenerator(unittest.TestCase):
    def test_get_tseries(self):
        ts_generator = SyntheticTSGenerator(12000, 10)
        ts_df = ts_generator.gen_tseries()
        self.assertEqual((12000, 10), ts_df.shape)

    def test_baseline(self):
        ts_generator = SyntheticTSGenerator(12000, 10)
        baseline = ts_generator.baseline()
        self.assertTrue(baseline)

    def test_trend(self):
        ts_generator = SyntheticTSGenerator(12000, 10)
        trend = ts_generator.trend()
        self.assertEqual((12000,), trend.shape)

    def test_seasonality(self):
        ts_generator = SyntheticTSGenerator(1000, 10)
        seasonal = ts_generator.seasonality(ts_generator.primary_period)
        self.assertEqual((1000,), seasonal.shape)

    def test_noise(self):
        ts_generator = SyntheticTSGenerator(12000, 10)
        noise = ts_generator.noise()
        self.assertEqual((12000,), noise.shape)

    def test_train_test_split(self):
        ts_generator = SyntheticTSGenerator(10080, 10)
        df = ts_generator.gen_tseries()
        train_df, test_df = ts_generator.train_test_split(df, 1440)
        self.assertEqual((8640, 10), train_df.shape)
        self.assertEqual((1440, 10), test_df.shape)


if __name__ == "__main__":
    unittest.main()
