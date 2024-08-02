import logging
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from xgboost import XGBRegressor, callback

from numalogic.tools.data import ForecastDataset
from numalogic.transforms._covariates import CovariatesGenerator

_LOGGER = logging.getLogger(__name__)


def _check_data_format(df) -> bool:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be of type pd.DataFrame")
    if not df.shape[1] > 0:
        raise ValueError("df should have more than 0 column")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index should be of type pd.DatetimeIndex")
    return True


class XGBoostForecaster:
    """
    A forecaster that uses XGBoost regressor to predict future values.

    Args:
    ____
        horizon: number of time steps to predict into the future
        seq_len: number of time steps to consider for prediction
        l_rate: learning rate for the XGBoost regress
        regressor_params: additional parameters for the XGBoost regressor
    """

    __slots__: ClassVar = [
        "horizon",
        "seq_len",
        "val_split",
        "model",
        "early_stop_callback",
        "early_stopping",
    ]

    def __init__(
        self,
        horizon: int,
        seq_len: int,
        regressor_params: Optional[dict] = None,
        early_stopping=True,
        val_split: float = 0.1,
    ):
        self.horizon = horizon
        self.seq_len = seq_len
        self.val_split: ClassVar = 0
        self.early_stopping = early_stopping
        if early_stopping:
            self.val_split = val_split
        early_stop_callback = callback.EarlyStopping(
            rounds=20, metric_name="rmse", save_best=True, maximize=False, min_delta=1e-4
        )
        default_params = {
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "booster": "gbtree",
            "max_depth": 7,
            "min_child_weight": 1,
            "gamma": 0.0,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "nthread": 4,
            "seed": 27,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        if early_stopping:
            default_params.update({"callbacks": [early_stop_callback]})
        if regressor_params:
            default_params.update(default_params)

        self.model: ClassVar = XGBRegressor(**default_params)

    def prepare_data(self, x: np.array):
        """
        Prepare data in the format required for forecasting.

        Args:
        ----
            x: np.array: input data
            seq_len: int: sequence length
            horizon: int: forecast horizon
        """
        ds = ForecastDataset(x, seq_len=self.seq_len, horizon=self.horizon)
        dataloaders = DataLoader(ds, batch_size=1)

        X = np.empty((0, self.seq_len, x.shape[1]))
        Y = np.empty((0, self.horizon, 1))
        for x, y in dataloaders:
            X = np.concatenate([X, x.numpy()], axis=0)
            Y = np.concatenate([Y, y[:, :, 0].unsqueeze(-1).numpy()], axis=0)
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)
        return X, Y

    def fit(self, df: pd.DataFrame):
        _check_data_format(df)

        # Split the data into training and validation sets
        train_df = df.iloc[: int(len(df) * (1 - self.val_split)), :]
        val_df = df.iloc[int(len(df) * (1 - self.val_split)) :, :] if self.val_split else None

        # Transform and prepare the training data
        transformed_train_data = CovariatesGenerator().transform(train_df)
        x_train, y_train = self.prepare_data(transformed_train_data)

        # Fit the model with or without validation data
        if val_df is not None:
            transformed_val_data = CovariatesGenerator().transform(val_df)
            x_val, y_val = self.prepare_data(transformed_val_data)
            _LOGGER.info("Fitting the model with validation data")
            self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True)
        else:
            _LOGGER.info("Fitting the model without validation data")
            self.model.fit(x_train, y_train, verbose=False)

    def predict_horizon(self, df: pd.DataFrame) -> np.ndarray:
        _check_data_format(df)
        transformed_test_data = CovariatesGenerator().transform(df)
        _LOGGER.info("Predicting the horizon")
        x_test, y_test = self.prepare_data(transformed_test_data)
        return self.model.predict(x_test)

    def predict_last(self, df: pd.DataFrame) -> np.ndarray:
        _check_data_format(df)
        transformed_test_data = CovariatesGenerator().transform(df)
        _LOGGER.info("Predicting the last value")
        x_test, y_test = self.prepare_data(transformed_test_data)
        return self.model.predict(x_test[-1].reshape(1, -1))

    def save_artifacts(self, path: str) -> None:
        artifact = {"model": self.model}
        torch.save(artifact, path)
        _LOGGER.info(f"Model saved at {path}")

    def load_artifacts(self, path: str) -> None:
        artifact = torch.load(path)
        self.model = artifact["model"]
        _LOGGER.info(f"Model loaded from {path}")
