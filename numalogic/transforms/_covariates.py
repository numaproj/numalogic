import logging
from typing import Union

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries

from numalogic.base import StatelessTransformer

_LOGGER = logging.getLogger(__name__)


class CovariatesGenerator(StatelessTransformer):
    """A transformer/generator that generates covariates for a timeseries dataset.

    Args:
    ----
        timestamp_column_name: The name of the timestamp column
        columns_to_preserve: The columns to preserve in the dataset
        *covariate_attributes: The tuple of attributes to consider for generating covariates
    """

    def __init__(
        self,
        timestamp_column_name: str = "timestamp",
        columns_to_preserve: Union[str, list[str]] = "value",
        *covariate_attributes: tuple[str],
    ):
        self.covariate_attributes = (
            covariate_attributes if covariate_attributes else ("dayofweek", "month", "dayofyear")
        )
        self.timestamp_column_name = timestamp_column_name
        self.columns_to_preserve = columns_to_preserve

    def _get_covariates(self, df: pd.DataFrame):
        covariates = []
        _LOGGER.info("Generating covariates for %s", self.covariate_attributes)
        for attribute in self.covariate_attributes:
            day_series = datetime_attribute_timeseries(
                TimeSeries.from_dataframe(
                    df.reset_index(), self.timestamp_column_name, self.columns_to_preserve
                ),
                attribute=attribute,
                one_hot=False,
                cyclic=True,
            ).values()
            covariates.append(day_series)
        return np.concatenate(covariates, axis=1)

    def transform(self, input_: pd.DataFrame, **__):
        """
        Generate covariates for a timeseries dataset.

        Args:
        ----
            data: np.array: input data
        """
        covariates = self._get_covariates(input_)
        return np.concatenate([input_.to_numpy(), covariates], axis=1)
