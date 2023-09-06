import logging
from datetime import datetime
from typing import Optional, Final

import numpy as np
import orjson
import pandas as pd
import requests
from orjson import JSONDecodeError

from numalogic.connectors._base import DataFetcher
from numalogic.tools.exceptions import PrometheusFetcherError, PrometheusInvalidResponseError

LOGGER = logging.getLogger(__name__)
MAX_DATA_POINTS: Final[int] = 11000
_MAX_RECURSION_DEPTH: Final[int] = 10
_API_ENDPOINT: Final[str] = "/api/v1/query_range"


class PrometheusFetcher(DataFetcher):
    """
    Class for fetching data as a dataframe from Prometheus or Thanos.

    Args:
        prometheus_server: Prometheus/Thanos URL
        scrape_interval_secs: Prometheus scrape interval in seconds
    """

    __slots__ = ("_endpoint", "_step_secs")

    def __init__(self, prometheus_server: str, scrape_interval_secs: int = 30):
        super().__init__(prometheus_server)
        self._endpoint = f"{self.url}{_API_ENDPOINT}"
        self._step_secs = scrape_interval_secs

    @staticmethod
    def build_query(metric: str, filters: dict[str, str]) -> str:
        """Builds a Prometheus query string from metric name and filters."""
        query = metric
        if filters:
            label_filters = ",".join(f"{k}='{v}'" for k, v in filters.items())
            return f"{metric}{{{label_filters}}}"
        return query

    def fetch_data(
        self,
        metric_name: str,
        start: datetime,
        end: Optional[datetime] = None,
        filters: Optional[dict[str, str]] = None,
        return_labels: Optional[list[str]] = None,
        aggregate: bool = True,
        fill_na_value: float = 0.0,
    ) -> pd.DataFrame:
        """
        Fetches data from Prometheus/Thanos and returns a dataframe.

        Args:
            metric_name: Prometheus metric name
            start: Start time
            end: End time
            filters: Prometheus label filters
            return_labels: Prometheus label names as columns to return
            aggregate: Whether to aggregate the data
            fill_na_value: Value to fill NaNs with

        Returns
        -------
            Dataframe with timestamp and metric values

        Raises
        ------
            ValueError: If end time is before start time
            PrometheusFetcherError: If there is an error while fetching data
            PrometheusInvalidResponseError: If the response from Prometheus is invalid
            RecursionError: If the recursive depth exceeds the max depth
        """
        query = self.build_query(metric_name, filters)

        LOGGER.debug("Prometheus Query: %s", query)

        if not end:
            end = datetime.now()
        end_ts = int(end.timestamp())
        start_ts = int(start.timestamp())

        if end_ts < start_ts:
            raise ValueError(f"end_time: {end} must not be before start_time: {start}")

        results = self.query_range(query, start_ts, end_ts)

        df = pd.json_normalize(results)
        return_labels = [f"metric.{label}" for label in return_labels or []]
        if df.empty:
            LOGGER.warning("Query returned no results")
            return df

        df = df[["values", *return_labels]]
        df = df.explode("values", ignore_index=True)
        df[["timestamp", metric_name]] = df["values"].to_list()
        df.drop(columns=["values"], inplace=True)
        df = df.astype({metric_name: float})

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.sort_values(by=["timestamp"], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(fill_na_value)

        if aggregate and return_labels:
            df = df.groupby(by=["timestamp"]).apply(lambda x: x[[metric_name]].mean())

        return df

    def _api_query_range(self, query: str, start_ts: int, end_ts: int) -> list[dict]:
        """Queries Prometheus API for data.""" ""
        try:
            response = requests.get(
                self._endpoint,
                params={
                    "query": query,
                    "start": start_ts,
                    "end": end_ts,
                    "step": f"{self._step_secs}s",
                },
            )
        except Exception as err:
            raise PrometheusFetcherError("Error while fetching data from Prometheus") from err

        try:
            results = orjson.loads(response.text)["data"]["result"]
        except (KeyError, JSONDecodeError) as err:
            LOGGER.exception("Invalid response from Prometheus: %s", response.text)
            raise PrometheusInvalidResponseError("Invalid response from Prometheus") from err
        return results

    def query_range(self, query: str, start_ts: int, end_ts: int, _depth=0) -> list[dict]:
        """
        Queries Prometheus API recursively for data.

        Args:
            query: Prometheus query string
            start_ts: Start timestamp
            end_ts: End timestamp
            _depth: Current recursion depth

        Returns
        -------
            List of Prometheus results

        Raises
        ------
            RecursionError: If the recursive depth exceeds the max depth
        """
        if _depth > _MAX_RECURSION_DEPTH:
            raise RecursionError(f"Max recursive depth of {_depth} reached")
        datapoints = (end_ts - start_ts) // self._step_secs

        if datapoints > MAX_DATA_POINTS:
            max_end_ts = start_ts + (MAX_DATA_POINTS * self._step_secs)
            return self.query_range(query, start_ts, max_end_ts, _depth + 1) + self.query_range(
                query, max_end_ts + self._step_secs, end_ts, _depth + 1
            )
        return self._api_query_range(query, start_ts, end_ts)
