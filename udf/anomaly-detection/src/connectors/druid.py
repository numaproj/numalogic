import time
from collections.abc import Hashable
from typing import Sequence

import pytz
import logging
import pandas as pd
from datetime import datetime, timedelta

from omegaconf import OmegaConf
from pydruid.client import PyDruid
from pydruid.utils.filters import Filter
import urllib.request
import urllib.error

from src.connectors._config import Pivot

_LOGGER = logging.getLogger(__name__)


class DruidFetcher:
    def __init__(self, url: str, endpoint: str):
        self.client = PyDruid(url, endpoint)

    def fetch_data(
            self,
            datasource: str,
            filter_keys: list[str],
            filter_values: list[str],
            dimensions: list[str],
            granularity: str = "minute",
            aggregations: dict = None,
            group_by: list[str] = None,
            pivot: Pivot = None,
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

        _LOGGER.info(
            "Fetching data with params: %s",
            params,
        )

        response = self.client.groupby(**params)
        df = response.export_pandas()

        if df is None or df.shape[0] == 0:
            logging.warning("No data found for keys %s", filter_pairs)
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10 ** 6

        if group_by:
            df = df.groupby(by=group_by).sum().reset_index()

        if pivot:
            df = df.pivot(
                index=pivot.index,
                columns=pivot.columns,
                values=pivot.value,
            )
            df.columns = df.columns.map('{0[1]}'.format)
            df.reset_index(inplace=True)

        _LOGGER.info(
            "Time taken to fetch data: %s, for keys: %s, for df shape: %s",
            time.time() - _start_time,
            filter_pairs,
            df.shape,
        )
        return df


