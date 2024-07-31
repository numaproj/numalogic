import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from numalogic.tools.data import ForecastDataset


############ helper functions ###############
def get_covariates(df: pd.DataFrame, attributes: list):
    """
    Get covariates for forecasting.

    Args:
    ----
        df: pd.DataFrame: input data
        attributes: list: list of attributes to be used as covariates
    """
    covariates = []
    for attribute in attributes:
        day_series = datetime_attribute_timeseries(
            TimeSeries.from_dataframe(df.reset_index(), "timestamp", "value"),
            attribute=attribute,
            one_hot=False,
            cyclic=True,
        ).values()

        covariates.append(day_series)

    return np.concatenate(covariates, axis=1)


def get_data(x: np.array, seq_len: int, horizon: int):
    """
    Get data in the format required for forecasting.

    Args:
    ----
        x: np.array: input data
        seq_len: int: sequence length
        horizon: int: forecast horizon
    """
    ds = ForecastDataset(x, seq_len=seq_len, horizon=horizon)
    dataloaders = DataLoader(ds, batch_size=1)

    X = np.empty((0, seq_len, x.shape[1]))
    Y = np.empty((0, horizon, 1))
    for x, y in dataloaders:
        X = np.concatenate([X, x.numpy()], axis=0)
        Y = np.concatenate([Y, y[:, :, 0].unsqueeze(-1).numpy()], axis=0)
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    return X, Y


#########################################


def preprocess_data(
    df: pd.DataFrame, seq_len: int, horizon: int, scaler: StandardScaler, fit: bool = False
):
    """
    Preprocess data for forecasting.

    Args:
    ----
        df: pd.DataFrame: input data
        seq_len: int: sequence length
        horizon: int: forecast horizon
        scaler: StandardScaler: scaler object
        fit: bool: whether to fit the scaler or not
    """
    # get covariates
    covariates = get_covariates(df, ["dayofweek", "hour", "dayofyear"])

    # standard scaling
    if fit:
        scaled_data = scaler.fit_transform(df.to_numpy())
    else:
        scaled_data = scaler.transform(df.to_numpy())
    # data in numpy format
    data = np.concatenate([scaled_data, covariates], axis=1)

    # data in final format
    x_data, y_data = get_data(data, seq_len, horizon)

    return x_data, y_data
