import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import pytz
from pydruid.client import PyDruid
from pydruid.utils.filters import Filter
from numalogic.connectors._config import Pivot
from typing import Optional

_LOGGER = logging.getLogger(__name__)


class DruidFetcher:
    """
    Class for fetching data as a dataframe from Druid.

    Args:
        url: Druid URL
        endpoint: Druid endpoint
    """

    def __init__(self, url: str, endpoint: str):
        self.client = PyDruid(url, endpoint)

    def fetch_data(
        self,
        datasource: str,
        filter_keys: list[str],
        filter_values: list[str],
        dimensions: list[str],
        granularity: str = "minute",
        aggregations: Optional[dict] = None,
        group_by: Optional[list[str]] = None,
        pivot: Optional[Pivot] = None,
        hours: float = 24,
    ) -> pd.DataFrame:
        _start_time = time.time()
        filter_pairs = {}
        for k, v in zip(filter_keys, filter_values):
            filter_pairs[k] = v

        _filter = Filter(
            type="and",
            fields=[Filter(type="selector", dimension=k, value=v) for k, v in filter_pairs.items()],
        )

        end_dt = datetime.now(pytz.utc)
        start_dt = end_dt - timedelta(hours=hours)
        intervals = f"{start_dt.isoformat()}/{end_dt.isoformat()}"

        params = {
            "datasource": datasource,
            "granularity": granularity,
            "intervals": intervals,
            "aggregations": aggregations,
            "filter": _filter,
            "dimensions": dimensions,
        }

        _LOGGER.debug(
            "Fetching data with params: %s",
            params,
        )

        response = self.client.groupby(**params)
        df = response.export_pandas()

        if df is None or df.shape[0] == 0:
            logging.warning("No data found for keys %s", filter_pairs)
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**6

        if group_by:
            df = df.groupby(by=group_by).sum().reset_index()

        if pivot.columns:
            df = df.pivot(
                index=pivot.index,
                columns=pivot.columns,
                values=pivot.value,
            )
            df.columns = df.columns.map("{0[1]}".format)
            df.reset_index(inplace=True)
        return df
