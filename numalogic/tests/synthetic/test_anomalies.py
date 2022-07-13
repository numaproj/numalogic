import unittest

from matplotlib import pyplot as plt

from numalogic.synthetic import AnomalyGenerator, SyntheticTSGenerator


class TestAnomalyGenerator(unittest.TestCase):
    def test_inject_global_anomalies(self, plot=False):
        ts_generator = SyntheticTSGenerator(7200, 5, seasonal_ts_prob=1.0)
        ts_df = ts_generator.gen_tseries()
        train_df, test_df = ts_generator.train_test_split(ts_df, 1440)

        anomaly_generator = AnomalyGenerator(train_df)
        cols = ["s1", "s2"]
        outlier_df = anomaly_generator.inject_anomalies(test_df, cols=cols)
        if plot:
            ax1 = plt.subplot(211)
            outlier_df[cols].plot(ax=ax1, title="Outlier")
            ax2 = plt.subplot(212)
            test_df[cols].plot(ax=ax2, title="Original")
        plt.show()
        self.assertEqual(test_df.shape, outlier_df.shape)
        other_cols = test_df.columns.difference(cols)
        self.assertTrue(test_df[other_cols].equals(outlier_df[other_cols]))
        self.assertFalse(test_df.equals(outlier_df))

    def test_inject_contextual_anomalies(self, plot=False):
        ts_generator = SyntheticTSGenerator(7200, 5, seasonal_ts_prob=1.0)
        ts_df = ts_generator.gen_tseries()
        train_df, test_df = ts_generator.train_test_split(ts_df, 1440)

        anomaly_generator = AnomalyGenerator(train_df, anomaly_type="contextual")
        cols = ["s1", "s2"]
        outlier_df = anomaly_generator.inject_anomalies(test_df, cols=cols)
        if plot:
            ax1 = plt.subplot(211)
            outlier_df[cols].plot(ax=ax1, title="Outlier")
            ax2 = plt.subplot(212)
            test_df[cols].plot(ax=ax2, title="Original")
        plt.show()
        self.assertEqual(test_df.shape, outlier_df.shape)
        self.assertFalse(test_df.equals(outlier_df))
        other_cols = test_df.columns.difference(cols)
        self.assertTrue(test_df[other_cols].equals(outlier_df[other_cols]))

    def test_inject_collective_anomalies(self, plot=False):
        ts_generator = SyntheticTSGenerator(7200, 5, seasonal_ts_prob=1.0)
        ts_df = ts_generator.gen_tseries()
        train_df, test_df = ts_generator.train_test_split(ts_df, 1440)

        anomaly_generator = AnomalyGenerator(train_df, anomaly_type="collective")
        cols = ["s1", "s2"]
        outlier_df = anomaly_generator.inject_anomalies(test_df, cols=cols)
        if plot:
            ax1 = plt.subplot(211)
            outlier_df[cols].plot(ax=ax1, title="Outlier")
            ax2 = plt.subplot(212)
            test_df[cols].plot(ax=ax2, title="Original")
        plt.show()
        self.assertEqual(test_df.shape, outlier_df.shape)
        self.assertFalse(test_df.equals(outlier_df))
        other_cols = test_df.columns.difference(cols)
        self.assertTrue(test_df[other_cols].equals(outlier_df[other_cols]))

    def test_inject_causal_anomalies(self, plot=False):
        ts_generator = SyntheticTSGenerator(7200, 5, seasonal_ts_prob=1.0)
        ts_df = ts_generator.gen_tseries()
        train_df, test_df = ts_generator.train_test_split(ts_df, 1440)

        anomaly_generator = AnomalyGenerator(train_df, anomaly_type="causal")
        cols = ["s1", "s2", "s3"]
        outlier_df = anomaly_generator.inject_anomalies(test_df, cols=cols)
        if plot:
            ax1 = plt.subplot(211)
            outlier_df[cols].plot(ax=ax1, title="Outlier")
            ax2 = plt.subplot(212)
            test_df[cols].plot(ax=ax2, title="Original")
        plt.show()
        self.assertEqual(test_df.shape, outlier_df.shape)
        self.assertFalse(test_df.equals(outlier_df))
        other_cols = test_df.columns.difference(cols)
        self.assertTrue(test_df[other_cols].equals(outlier_df[other_cols]))

    def test_injected_cols(self):
        ts_generator = SyntheticTSGenerator(7200, 5, seasonal_ts_prob=1.0)
        ts_df = ts_generator.gen_tseries()
        train_df, test_df = ts_generator.train_test_split(ts_df, 1440)

        anomaly_generator = AnomalyGenerator(train_df)
        cols = ["s1", "s2"]
        anomaly_generator.inject_anomalies(test_df, cols=cols)
        self.assertListEqual(cols, anomaly_generator.injected_cols)

    def test_injected_no_cols(self):
        ts_generator = SyntheticTSGenerator(7200, 5, seasonal_ts_prob=1.0)
        ts_df = ts_generator.gen_tseries()
        train_df, test_df = ts_generator.train_test_split(ts_df, 1440)

        anomaly_generator = AnomalyGenerator(train_df, anomaly_type="causal")
        outlier_df = anomaly_generator.inject_anomalies(test_df)

        self.assertEqual(test_df.shape, outlier_df.shape)
        self.assertFalse(test_df.equals(outlier_df))
        self.assertEqual(2, len(anomaly_generator.injected_cols))

    def test_invalid_anomaly_type(self):
        ts_generator = SyntheticTSGenerator(7200, 5, seasonal_ts_prob=1.0)
        ts_df = ts_generator.gen_tseries()
        train_df, test_df = ts_generator.train_test_split(ts_df, 1440)

        anomaly_generator = AnomalyGenerator(train_df, anomaly_type="Hahaha")
        with self.assertRaises(AttributeError):
            anomaly_generator.inject_anomalies(test_df)


if __name__ == "__main__":
    unittest.main()
