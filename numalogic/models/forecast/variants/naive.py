import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBRegressor, callback

from numalogic.models.forecast.variants.utility import preprocess_data
from numalogic.transforms._postprocess import tanh_norm


class BaselineForecaster:
    """A baseline forecaster that uses a combination of:
    1. log transform
    2. Z score normalization.
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

    def find_threshold(self, train_df: pd.DataFrame, k=3) -> dict[str, tuple[float, float]]:
        for col in train_df.columns:
            mean = train_df[col].mean()
            std = max(1e-2, train_df[col].std())
            thresh_upper = mean + (k * std)
            thresh_lower = mean - (k * std)
            self.thresholds[col] = (thresh_lower, thresh_upper)
            self.means[col] = mean
        return self.thresholds

    def fit(self, train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
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
    """A simple forecaster that predicts the previous day/week values."""

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


class XGBoostForecaster:
    """
    A forecaster that uses XGBoost regressor to predict future values.

    Args:
    ____
        horizon: number of time steps to predict into the future
        seq_len: number of time steps to consider for prediction
        l_rate: learning rate for the XGBoost regress
    """

    def __is_standard_scaler_fitted(self) -> bool:
        return hasattr(self.scaler, "mean_")

    def __init__(self, horizon, seq_len, l_rate=0.1):
        self.horizon = horizon
        self.seq_len = seq_len
        self.learning_rate = l_rate
        self.scaler = StandardScaler()
        early_stop_callback = callback.EarlyStopping(
            rounds=20, metric_name="rmse", save_best=True, maximize=False, min_delta=1e-4
        )

        self.model = XGBRegressor(
            learning_rate=self.learning_rate,
            n_estimators=1000,
            booster="gbtree",
            max_depth=7,
            min_child_weight=1,
            gamma=0.0,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            nthread=4,
            seed=27,
            objective="reg:squarederror",
            random_state=42,  # eval_metric = 'mae',
            # early_stopping_rounds=20,
            callbacks=[early_stop_callback],
        )

    def fit(self, df: pd.DataFrame):
        train_df = df.iloc[: int(len(df) * 0.9), :]
        val_df = df.iloc[int(len(df) * 0.9) :, :]
        (
            x_train,
            y_train,
        ) = preprocess_data(
            df=train_df, scaler=self.scaler, seq_len=self.seq_len, horizon=self.horizon, fit=True
        )
        (
            x_val,
            y_val,
        ) = preprocess_data(
            df=val_df, scaler=self.scaler, seq_len=self.seq_len, horizon=self.horizon
        )
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.__is_standard_scaler_fitted():
            raise ValueError(
                "Standard Scaler not fitted. Please fit the model first or load the artifacts first"
            )
        (
            x_test,
            y_test,
        ) = preprocess_data(df=df, scaler=self.scaler, seq_len=self.seq_len, horizon=self.horizon)
        return self.model.predict(x_test)

    def save_artifacts(self, path: str) -> None:
        artifact = {"model": self.model, "scaler": self.scaler}
        torch.save(artifact, path)

    def load_artifacts(self, path: str) -> None:
        artifact = torch.load(path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
