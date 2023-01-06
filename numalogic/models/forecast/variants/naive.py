from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from numalogic.postprocess import tanh_norm


class BaselineForecaster:
    """
    A baseline forecaster that uses a combination of:
    1. log transform
    2. Z score normalization
    """

    def __init__(self):
        self.thresholds = {}
        transform = FunctionTransformer(np.log1p)
        z_scaler = StandardScaler()
        self.pipeline = make_pipeline(transform, z_scaler)
        self.means = {}

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.pipeline.transform(df), index=df.index, columns=df.columns)

    def inverse_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.pipeline.inverse_transform(df), index=df.index, columns=df.columns)

    def find_threshold(self, train_df: pd.DataFrame, k=3) -> Dict[str, Tuple[float, float]]:
        for col in train_df.columns:
            mean = train_df[col].mean()
            std = max(1e-2, train_df[col].std())
            thresh_upper = mean + (k * std)
            thresh_lower = mean - (k * std)
            self.thresholds[col] = (thresh_lower, thresh_upper)
            self.means[col] = mean
        return self.thresholds

    def fit(self, train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        self.pipeline.fit(train_df)
        scaled_train_df = self.normalize(train_df)
        return self.find_threshold(scaled_train_df)

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        _all_series = {}
        for feature in self.means:
            _all_series[feature] = pd.Series(
                np.full(test_df.shape[0], fill_value=self.means[feature])
            )
        return self.inverse_normalize(pd.DataFrame(_all_series))

    def score(self, test_df: pd.DataFrame) -> pd.DataFrame:
        scaled_test_df = self.normalize(test_df)
        anomaly_vecs = {}

        for col in scaled_test_df.columns:
            deviation = abs(scaled_test_df[col])
            thresh_vec = np.full(deviation.shape[0], self.thresholds[col][1])
            raw_anomaly_score = deviation / thresh_vec
            anomaly_vecs[col] = tanh_norm(raw_anomaly_score)
        return pd.DataFrame(anomaly_vecs, index=scaled_test_df.index)

    def r2_score(self, test_df: pd.DataFrame, multioutput="uniform_average") -> float:
        pred_df = self.predict(test_df)
        return r2_score(test_df, pred_df, multioutput=multioutput)


class SeasonalNaiveForecaster:
    """
    A simple forecaster that predicts the previous day/week values.
    """

    def __init__(self, season="daily"):
        self.pipeline = StandardScaler()
        self.norm_train_df = None
        if season == "daily":
            self.period = 1440
        elif season == "weekly":
            self.period = 10080
        else:
            raise NotImplementedError()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.pipeline.transform(df), index=df.index, columns=df.columns)

    def inverse_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.pipeline.inverse_transform(df), index=df.index, columns=df.columns)

    def fit(self, train_df: pd.DataFrame) -> None:
        if self.period > train_df.shape[0]:
            raise ValueError(f"Training set too small for period: {self.period}")
        self.pipeline.fit(train_df)
        self.norm_train_df = self.normalize(train_df)

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        test_size = test_df.shape[0]

        if test_size < self.period:
            pred_df = self.norm_train_df[-self.period : (-self.period + test_size)]
        elif test_size == self.period:
            pred_df = self.norm_train_df[-self.period :]
        else:
            raise RuntimeError("Cannot use Naive Forecaster for testsize > period")

        pred_df.index = test_df.index
        return self.inverse_normalize(pred_df)

    def r2_score(self, test_df: pd.DataFrame, multioutput="uniform_average") -> float:
        pred_df = self.predict(test_df)
        return r2_score(test_df, pred_df, multioutput=multioutput)
