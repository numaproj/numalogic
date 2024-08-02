import os

import numpy as np

from numalogic.models.forecast.variants import XGBoostForecaster
from numalogic.synthetic import SyntheticTSGenerator


def test_XGBForecaster1():
    ts_generator = SyntheticTSGenerator(seq_len=7200, num_series=1, freq="T")
    ts_df = ts_generator.gen_tseries()
    ts_df.columns = ["value"]
    ts_df.index.name = "timestamp"
    forecaster = XGBoostForecaster(horizon=6, seq_len=60)
    forecaster.fit(ts_df)
    forecaster.save_artifacts("model.pkl")
    forecaster_dummy = XGBoostForecaster(horizon=6, seq_len=60)
    forecaster_dummy.load_artifacts("model.pkl")
    os.remove("model.pkl")
    pred_1 = forecaster_dummy.predict_horizon(ts_df)
    pred_2 = forecaster.predict_horizon(ts_df)
    assert np.array_equal(pred_1, pred_2)


def test_XGBForecaster2():
    ts_generator = SyntheticTSGenerator(seq_len=7200, num_series=1, freq="T")
    ts_df = ts_generator.gen_tseries()
    ts_df.columns = ["value"]
    ts_df.index.name = "timestamp"
    forecaster = XGBoostForecaster(horizon=6, seq_len=60, early_stopping=False)
    forecaster.fit(ts_df)
    forecaster.save_artifacts("model.pkl")
    forecaster_dummy = XGBoostForecaster(horizon=6, seq_len=60)
    forecaster_dummy.load_artifacts("model.pkl")
    os.remove("model.pkl")
    pred_1 = forecaster_dummy.predict_last(ts_df)
    pred_2 = forecaster.predict_last(ts_df)
    assert np.array_equal(pred_1, pred_2)
