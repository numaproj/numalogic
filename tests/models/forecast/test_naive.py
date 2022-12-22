import unittest

from numalogic.synthetic.timeseries import SyntheticTSGenerator
from numalogic.models.forecast.variants import BaselineForecaster, SeasonalNaiveForecaster


class TestBaselineForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ts_generator = SyntheticTSGenerator(seq_len=7200, num_series=3, freq="T")
        ts_df = ts_generator.gen_tseries()
        cls.train_df, cls.test_df = ts_generator.train_test_split(ts_df, test_size=1440)

    def test_predict(self):
        model = BaselineForecaster()
        model.fit(self.train_df)
        pred_df = model.predict(self.test_df)
        self.assertEqual(pred_df.shape, self.test_df.shape)

    def test_scores(self):
        model = BaselineForecaster()
        model.fit(self.train_df)
        pred_df = model.predict(self.test_df)
        r2_score = model.r2_score(self.test_df)
        anomaly_df = model.score(self.test_df)

        self.assertIsInstance(r2_score, float)
        self.assertEqual(pred_df.shape, self.test_df.shape)
        self.assertEqual(anomaly_df.shape, self.test_df.shape)


class TestSeasonalNaiveForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ts_generator = SyntheticTSGenerator(seq_len=7200, num_series=3, freq="T")
        ts_df = ts_generator.gen_tseries()
        cls.train_df, cls.test_df = ts_generator.train_test_split(ts_df, test_size=1440)

    def test_predict(self):
        model = SeasonalNaiveForecaster()
        model.fit(self.train_df)
        pred_df = model.predict(self.test_df)
        self.assertEqual(pred_df.shape, self.test_df.shape)

    def test_scores(self):
        model = SeasonalNaiveForecaster()
        model.fit(self.train_df)
        pred_df = model.predict(self.test_df)
        r2_score = model.r2_score(self.test_df)

        self.assertEqual(self.test_df.shape, pred_df.shape)
        self.assertIsInstance(r2_score, float)

    def test_period_err_01(self):
        model = SeasonalNaiveForecaster(season="weekly")
        with self.assertRaises(ValueError):
            model.fit(self.train_df)

    def test_period_err_02(self):
        with self.assertRaises(NotImplementedError):
            SeasonalNaiveForecaster(season="yearly")

    def test_evalset_err(self):
        model = SeasonalNaiveForecaster()
        model.fit(self.train_df)
        with self.assertRaises(RuntimeError):
            model.predict(self.train_df)


if __name__ == "__main__":
    unittest.main()
