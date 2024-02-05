import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import pytz
from pydruid.client import PyDruid
from pydruid.utils.dimensions import DimensionSpec
from pydruid.utils.filters import Filter

from numalogic.connectors._base import DataFetcher
from numalogic.connectors._config import Pivot
from typing import Optional

from numalogic.tools.exceptions import DruidFetcherError

_LOGGER = logging.getLogger(__name__)
TIMEOUT = 10000


# TODO: pass dictionary of keys and values as dict
def make_filter_pairs(filter_keys: list[str], filter_values: list[str]) -> dict[str, str]:
    """

    Args:
        filter_keys: keys
        filter_values: values.

    Returns: a dict of key value pairs

    """
    return dict(zip(filter_keys, filter_values))


def build_params(
    datasource: str,
    dimensions: list[str],
    filter_pairs: dict,
    granularity: str,
    hours: float,
    delay: float,
    aggregations: Optional[list[str]] = None,
    post_aggregations: Optional[list[str]] = None,
) -> dict:
    """

    Args:
        datasource: Data source to query
        dimensions: The dimensions to group by
        filter_pairs: Indicates which rows of
          data to include in the query
        granularity: Time bucket to aggregate data by hour, day, minute, etc.,
        hours: Hours from now to skip training.
        delay: Added delay to the fetch query from current time.
        aggregations: A map from aggregator name to one of the
          ``pydruid.utils.aggregators`` e.g., ``doublesum``
        post_aggregations: A map from post aggregator name to one of the
          ``pydruid.utils.postaggregator`` e.g., ``QuantilesDoublesSketchToQuantile``.

    Returns: a dict of parameters

    """
    _filter = Filter(
        type="and",
        fields=[Filter(type="selector", dimension=k, value=v) for k, v in filter_pairs.items()],
    )
    end_dt = datetime.now(pytz.utc) - timedelta(hours=delay)
    _LOGGER.debug("Querying with end_dt: %s, that is with delay of %s hrs", end_dt, delay)

    start_dt = end_dt - timedelta(hours=hours)

    intervals = [f"{start_dt.isoformat()}/{end_dt.isoformat()}"]
    dimension_specs = map(lambda d: DimensionSpec(dimension=d, output_name=d), dimensions)

    return {
        "datasource": datasource,
        "granularity": granularity,
        "intervals": intervals,
        "aggregations": aggregations or dict(),
        "post_aggregations": post_aggregations or dict(),
        "filter": _filter,
        "dimensions": dimension_specs,
        "context": {"timeout": TIMEOUT, "configIds": list(filter_pairs), "source": "numalogic"},
    }


class DruidFetcher(DataFetcher):
    """
    Class for fetching data as a dataframe from Druid.

    Args:
        url: Druid URL
        endpoint: Druid endpoint

    Raises
    ------
        DruidFetcherError: If error/exception during fetching of data.

    """

    def __init__(self, url: str, endpoint: str):
        super().__init__(url)
        self.client = PyDruid(url, endpoint)

    def fetch(
        self,
        datasource: str,
        filter_keys: list[str],
        filter_values: list[str],
        dimensions: list[str],
        delay: float = 3.0,
        granularity: str = "minute",
        aggregations: Optional[dict] = None,
        post_aggregations: Optional[dict] = None,
        group_by: Optional[list[str]] = None,
        pivot: Optional[Pivot] = None,
        hours: float = 24,
    ) -> pd.DataFrame:
        _start_time = time.perf_counter()
        filter_pairs = make_filter_pairs(filter_keys, filter_values)
        query_params = build_params(
            datasource=datasource,
            dimensions=dimensions,
            filter_pairs=filter_pairs,
            granularity=granularity,
            hours=hours,
            delay=delay,
            aggregations=aggregations,
            post_aggregations=post_aggregations,
        )
        try:
            response = self.client.groupby(**query_params)
        except Exception as err:
            raise DruidFetcherError("Druid Exception:\n") from err
        else:
            df = response.export_pandas()
            if df.empty or df.shape[0] == 0:
                logging.warning("No data found for keys %s", filter_pairs)
                return pd.DataFrame()

            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**6

            if group_by:
                df = df.groupby(by=group_by).sum().reset_index()

            if pivot and pivot.columns:
                df = df.pivot(
                    index=pivot.index,
                    columns=pivot.columns,
                    values=pivot.value,
                )
                df.columns = df.columns.map("{0[1]}".format)
                df.reset_index(inplace=True)

            _end_time = time.perf_counter() - _start_time
            _LOGGER.debug("params: %s latency: %.6fs", query_params, _end_time)
            return df

    def raw_fetch(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError
