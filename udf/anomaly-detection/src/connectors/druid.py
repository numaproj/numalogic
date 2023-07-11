import pytz
import logging
import pandas as pd
from datetime import datetime, timedelta
from pydruid.client import PyDruid
from pydruid.query import QueryBuilder
from pydruid.utils.filters import Filter
import urllib.request
import urllib.error

_LOGGER = logging.getLogger(__name__)


class DruidFetcher:
    def __init__(self, url: str, endpoint: str):
        self.client = PyDruid(url, endpoint)
        self.query_builder = QueryBuilder()

    def fetch_data(
            self,
            datasource: str,
            filter_keys: list[str],
            filter_values: list[str],
            dimensions: list[str],
            granularity: str = "minute",
            aggregations: dict = None,
            group_by: list[str] = None,
            pivot: dict = None,
            hours: float = 24,
    ) -> pd.DataFrame:
        filter_pairs = {}
        for k, v in zip(filter_keys, filter_values):
            filter_pairs[k] = v

        _filter = Filter(
            type="and",
            fields=[
                Filter(type="selector", dimension=k, value=v)
                for k, v in filter_pairs.items()
            ],
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
        response = self.client.groupby(**params)
        df = response.export_pandas()

        if df is None or df.shape[0] == 0:
            logging.warning("No data found for keys %s", filter_pairs)
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp']).astype("int64") // 10 ** 6

        if group_by:
            df = df.groupby(by=group_by).sum().reset_index()

        if pivot:
            df = df.pivot(
                index=pivot["index"],
                columns=pivot["columns"],
                values=pivot["values"],
            )

        return df


# url = "https://getafix.odldruid-prd.a.intuit.com"
# endpoint = "druid/v2/"
#
# druid_fetcher = DruidFetcher(url, endpoint)
# df = druid_fetcher.fetch_data(
#     filter_keys=["assetId"],
#     filter_values=["1084259202722926969"],
#     dimensions=["ciStatus"],
#     datasource="tech-ip-customer-interaction-metrics",
#     aggregations={"count": doublesum("count")},
#     group_by=["timestamp", "ciStatus"],
#     hours=0.5,
#     pivot={
#         "index": "timestamp",
#         "columns": ["ciStatus"],
#         "values": "count"
#     }
# )
#
# print(df)
