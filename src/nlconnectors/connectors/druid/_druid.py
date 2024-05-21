import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import pytz
from pydruid.client import PyDruid
from pydruid.utils.dimensions import DimensionSpec
from pydruid.utils.filters import Filter

from numalogic.connectors._base import DataFetcher
from numalogic.connectors._config import Pivot, FilterConf
from typing import Optional, Final

from numalogic.tools.exceptions import DruidFetcherError

_LOGGER = logging.getLogger(__name__)
TIMEOUT: Final[int] = 10000
_MAX_CONCURRENCY: Final[int] = 16


# TODO: pass dictionary of keys and values as dict
def make_filter_pairs(filter_keys: list[str], filter_values: list[str]) -> dict[str, str]:
    """

    Args:
        filter_keys: keys
        filter_values: values.

    Returns: a dict of key value pairs

    """
    return dict(zip(filter_keys, filter_values))


def _combine_in_filters(filters_list) -> Filter:
    return Filter(type="and", fields=[Filter(**item) for item in filters_list])


def _combine_ex_filters(filters_list) -> Filter:
    filters = _combine_in_filters(filters_list)
    return Filter(type="not", field=filters)


def _make_static_filters(filters: FilterConf) -> Filter:
    filter_list = []
    if filters.inclusion_filters:
        filter_list.append(_combine_in_filters(filters.inclusion_filters))
    if filters.exclusion_filters:
        filter_list.append(_combine_ex_filters(filters.exclusion_filters))
    return Filter(type="and", fields=filter_list)


def build_params(
    datasource: str,
    dimensions: list[str],
    filter_pairs: dict,
    granularity: str,
    hours: float,
    delay: float,
    static_filters: Optional[FilterConf] = None,
    aggregations: Optional[list[str]] = None,
    post_aggregations: Optional[list[str]] = None,
    reference_dt: Optional[datetime] = None,
) -> dict:
    """

    Args:
        datasource: Data source to query
        dimensions: The dimensions to group by
        filter_pairs: Indicates which rows of
          data to include in the query
        static_filters: Static filters passed from config
        granularity: Time bucket to aggregate data by hour, day, minute, etc.,
        hours: Hours from now to skip training.
        delay: Added delay to the fetch query from current time.
        aggregations: A map from aggregator name to one of the
          ``pydruid.utils.aggregators`` e.g., ``doublesum``
        post_aggregations: A map from post aggregator name to one of the
          ``pydruid.utils.postaggregator`` e.g., ``QuantilesDoublesSketchToQuantile``.
        reference_dt: reference datetime to calculate start and end dt
            (None will mean using current datetime).

    Returns: a dict of parameters

    """
    _filter = Filter(
        type="and",
        fields=[Filter(type="selector", dimension=k, value=v) for k, v in filter_pairs.items()],
    )
    if static_filters:
        _LOGGER.debug("Static Filters are present!")
        _static_filters = _make_static_filters(static_filters)
        _filter = Filter(type="and", fields=[_static_filters, _filter])

    reference_dt = reference_dt or datetime.now(pytz.utc)
    end_dt = reference_dt - timedelta(hours=delay)
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

    __slots__ = ("client",)

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
        static_filters: Optional[FilterConf] = None,
        aggregations: Optional[dict] = None,
        post_aggregations: Optional[dict] = None,
        group_by: Optional[list[str]] = None,
        pivot: Optional[Pivot] = None,
        hours: float = 24,
    ) -> pd.DataFrame:
        """
        Fetch data from Druid.

        Args:
        ------
            datasource: Data source to query
            filter_keys: keys
            filter_values: values
            dimensions: The dimensions to group by
            delay: Added delay to the fetch query from current time.
            granularity: Time bucket to aggregate data by hour, day, minute, etc.
            static_filters: user defined filters
            aggregations: A map from aggregator name to one of the
                ``pydruid.utils.aggregators`` e.g., ``doublesum``
            post_aggregations: postaggregations map
            group_by: List of columns to group by
            pivot: Pivot configuration
            hours: Hours from now to fetch.

        Returns
        -------
            Fetched dataframe
        """
        _start_time = time.perf_counter()
        filter_pairs = make_filter_pairs(filter_keys, filter_values)
        query_params = build_params(
            datasource=datasource,
            dimensions=dimensions,
            filter_pairs=filter_pairs,
            static_filters=static_filters,
            granularity=granularity,
            hours=hours,
            delay=delay,
            aggregations=aggregations,
            post_aggregations=post_aggregations,
        )
        df = self._fetch(**query_params)

        if df.empty or df.shape[0] == 0:
            logging.warning("No data found for keys %s", filter_pairs)
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**6

        if group_by:
            df = df.groupby(by=group_by).sum().reset_index()

        # TODO: performance review
        if pivot:
            pivoted_frames = []
            for idx, column in enumerate(pivot.columns):
                _df = df.pivot_table(
                    index=pivot.index, columns=[column], values=pivot.value, aggfunc=pivot.agg[idx]
                )
                pivoted_frames.append(_df)

            df = pd.concat(pivoted_frames, axis=1, join="outer")
            df.columns = df.columns.map("{0[1]}".format)
            df.reset_index(inplace=True)

        _end_time = time.perf_counter() - _start_time
        _LOGGER.debug("Druid params: %s, fetch time: %.4fs", query_params, _end_time)
        return df

    def raw_fetch(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def chunked_fetch(
        self,
        datasource: str,
        filter_keys: list[str],
        filter_values: list[str],
        dimensions: list[str],
        delay: float = 3.0,
        granularity: str = "minute",
        static_filter: Optional[FilterConf] = None,
        aggregations: Optional[dict] = None,
        post_aggregations: Optional[dict] = None,
        group_by: Optional[list[str]] = None,
        pivot: Optional[Pivot] = None,
        hours: int = 24,
        chunked_hours: int = 6,
    ) -> pd.DataFrame:
        """
        Fetch data concurrently, and concatenate the results.

        Args:
        ------
            datasource: Data source to query
            filter_keys: keys
            filter_values: values
            dimensions: The dimensions to group by
            delay: Added delay to the fetch query from current time.
            granularity: Time bucket to aggregate data by hour, day, minute, etc.
            aggregations: A map from aggregator name to one of the
                ``pydruid.utils.aggregators`` e.g., ``doublesum``
            static_filter: user defined filters
            post_aggregations: postaggregations map
            group_by: List of columns to group by
            pivot: Pivot configuration
            hours: Hours from now to skip training.
            chunked_hours: Hours to fetch in each chunk

        Returns
        -------
            Fetched dataframe

        Raises
        ------
            ValueError: If chunked_hours is less than 1
        """
        if chunked_hours < 1:
            raise ValueError("chunked_hours should be integer and >= 1.")

        _start_time = time.perf_counter()
        filter_pairs = make_filter_pairs(filter_keys, filter_values)

        hours_elapsed = 0
        chunked_dfs = []
        qparams = []
        curr_time = datetime.now(pytz.utc)

        while hours_elapsed < hours:
            ref_dt = curr_time - timedelta(hours=hours_elapsed)
            qparams.append(
                build_params(
                    datasource=datasource,
                    dimensions=dimensions,
                    filter_pairs=filter_pairs,
                    static_filters=static_filter,
                    granularity=granularity,
                    hours=min(chunked_hours, hours - hours_elapsed),
                    delay=delay,
                    aggregations=aggregations,
                    post_aggregations=post_aggregations,
                    reference_dt=ref_dt,
                )
            )
            hours_elapsed += chunked_hours

        max_threads = min(_MAX_CONCURRENCY, len(qparams))
        _LOGGER.debug("Fetching data concurrently with %s threads", max_threads)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(self._fetch, **params) for params in qparams]
            chunked_dfs.extend(future.result() for future in futures)
        df = pd.concat(chunked_dfs, axis=0, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**6

        if group_by:
            df = df.groupby(by=group_by).sum().reset_index()

        if pivot:
            pivoted_frames = []
            for idx, column in enumerate(pivot.columns):
                _df = df.pivot_table(
                    index=pivot.index, columns=[column], values=pivot.value, aggfunc=pivot.agg[idx]
                )
                pivoted_frames.append(_df)

            df = pd.concat(pivoted_frames, axis=1, join="outer")
            df.columns = df.columns.map("{0[1]}".format)
            df.reset_index(inplace=True)

        _LOGGER.debug("Fetch time: %.4fs", time.perf_counter() - _start_time)
        return df

    def _fetch(self, **query_params) -> pd.DataFrame:
        try:
            response = self.client.groupby(**query_params)
        except Exception as err:
            raise DruidFetcherError("Druid Exception:\n") from err
        return response.export_pandas()
