# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import reduce
from operator import iconcat
from typing import Optional, Final

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
_METRIC_KEY: Final[str] = "metric.__name__"


class PrometheusFetcher(DataFetcher):
    """
    Class for fetching data as a dataframe from Prometheus or Thanos.

    Args:
        prometheus_server: Prometheus/Thanos URL
        scrape_interval_secs: Prometheus scrape interval in seconds
    Raises:
        PrometheusFetcherError: If error/exception during fetching of data.
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

    def fetch(
        self,
        start: datetime,
        end: Optional[datetime] = None,
        metric_name: str = "",
        filters: Optional[dict[str, str]] = None,
        return_labels: Optional[list[str]] = None,
        aggregate: bool = True,
    ) -> pd.DataFrame:
        """
        Fetches data from Prometheus/Thanos and returns a dataframe.

        Args:
        -------
            start: Start time
            end: End time
            metric_name: Prometheus metric name (default="")
            filters: Prometheus label filters
            return_labels: Prometheus label names as columns to return
            aggregate: Whether to aggregate the data

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

        LOGGER.info("Constructed Prometheus Query: %s", query)

        end_ts, start_ts = self._init_startend_ts(end, start)
        results = self.query_range(query, start_ts, end_ts)

        df = pd.json_normalize(results)
        if df.empty:
            LOGGER.warning("Query returned no results")
            return df

        extra_labels = [f"metric.{label}" for label in return_labels or []]
        if metric_name:
            metric_names = [metric_name]
        else:
            metric_names = self._extract_metric_names(df)

        df.set_index(_METRIC_KEY, inplace=True)

        dfs = []
        for metric_name in metric_names:
            _df = self._consolidate_df(df.loc[[metric_name]], metric_name, extra_labels)
            dfs.append(_df.set_index(["timestamp", *extra_labels]))

        df = dfs[0].join(dfs[1:]).reset_index().set_index("timestamp")

        if return_labels:
            df.rename(columns=dict(zip(extra_labels, return_labels)), inplace=True)

        if aggregate:
            df = self._agg_df(df, metric_names)

        return df.sort_values(by=["timestamp"])

    def raw_fetch(
        self,
        query: str,
        start: datetime,
        end: Optional[datetime] = None,
        return_labels: Optional[list[str]] = None,
        aggregate: bool = True,
    ):
        """
        Fetches data from Prometheus/Thanos using the provided raw query and returns a dataframe.

        Args:
        -------
            query: Raw prometheus query
            start: Start time
            end: End time
            return_labels: Prometheus label names as columns to return
            aggregate: Whether to aggregate the data over each timestamp

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
        end_ts, start_ts = self._init_startend_ts(end, start)
        results = self.query_range(query, start_ts, end_ts)

        df = pd.json_normalize(results)
        if df.empty:
            LOGGER.warning("Query returned no results")
            return df

        extra_labels = [f"metric.{label}" for label in return_labels or []]
        metric_names = self._extract_metric_names(df)

        if metric_names is None:
            raise PrometheusInvalidResponseError("No metric names were extracted from the query")

        df.set_index(_METRIC_KEY, inplace=True)

        dfs = []
        for metric_name in metric_names:
            _df = self._consolidate_df(df.loc[[metric_name]], metric_name, extra_labels)
            _df.set_index(["timestamp", *extra_labels], inplace=True)
            dfs.append(_df)

        df = dfs[0].join(dfs[1:])
        df.reset_index(inplace=True)
        df.set_index("timestamp", inplace=True)

        if return_labels:
            df.rename(columns=dict(zip(extra_labels, return_labels)), inplace=True)

        if aggregate:
            df = self._agg_df(df, metric_names)

        df.sort_values(by=["timestamp"], inplace=True)
        return df

    @staticmethod
    def _agg_df(df, metric_names: list[str]) -> pd.DataFrame:
        return df.groupby(by=["timestamp"]).apply(lambda x: x[metric_names].mean())

    @staticmethod
    def _consolidate_df(df: pd.DataFrame, metric_name: str, return_labels: list[str]):
        df = df[["values", *return_labels]]
        df = df.explode("values", ignore_index=True)
        df[["timestamp", metric_name]] = df["values"].to_list()
        df.drop(columns=["values"], inplace=True)
        df = df.astype({metric_name: float})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    @staticmethod
    def _init_startend_ts(end: datetime, start: datetime) -> tuple[int, int]:
        if not end:
            end = datetime.now()
        end_ts = int(end.timestamp())
        start_ts = int(start.timestamp())
        if end_ts < start_ts:
            raise ValueError(f"end_time: {end} must not be before start_time: {start}")
        return end_ts, start_ts

    @staticmethod
    def _extract_metric_names(df: pd.DataFrame) -> Optional[list[str]]:
        try:
            metric_name = df["metric.__name__"].item()
        except ValueError:
            return df["metric.__name__"].unique()
        except KeyError:
            return None
        return [metric_name]

    def _api_query_range(self, query: str, start_ts: int, end_ts: int) -> list[dict]:
        """Queries Prometheus API for data."""
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
        -------
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

            with ThreadPoolExecutor(max_workers=2) as _executor:
                results = _executor.map(
                    self.query_range,
                    (query, query),
                    (start_ts, max_end_ts + self._step_secs),
                    (max_end_ts, end_ts),
                    (_depth + 1, _depth + 1),
                )
                return reduce(iconcat, results, [])
        return self._api_query_range(query, start_ts, end_ts)
