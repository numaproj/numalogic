import time
from datetime import datetime, timedelta

import pytz
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from src import get_logger

_LOGGER = get_logger(__name__)


class Prometheus:
    def __init__(self, prometheus_server: str):
        self.PROMETHEUS_SERVER = prometheus_server

    def query_metric(
        self,
        metric_name: str,
        start: float,
        end: float,
        labels_map: Dict = None,
        return_labels: List[str] = None,
        step: int = 30,
    ) -> pd.DataFrame:
        query = metric_name
        if labels_map:
            label_list = [str(key + "=" + "'" + labels_map[key] + "'") for key in labels_map]
            query = metric_name + "{" + ",".join(label_list) + "}"

        _LOGGER.debug("Prometheus Query: %s", query)

        if end < start:
            raise ValueError("end_time must not be before start_time")

        result = self.query_range(query, start, end, step)

        arr = np.array(result["values"])
        df = pd.DataFrame(arr, columns=["timestamp", metric_name])
        df = df.astype(float)

        data = result["metric"]
        if return_labels:
            for label in return_labels:
                if label in data:
                    df[label] = data[label]

        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index.astype(int), unit="s")
        return df

    def query_range(self, query: str, start: float, end: float, step: int = 30) -> Optional[Dict]:
        results = {}
        data_points = (end - start) / step
        temp_start = start
        while data_points > 11000:
            temp_end = temp_start + 11000 * step
            response = self.query_range_limit(query, temp_start, temp_end, step)
            if results:
                results["values"] = results["values"] + response["values"]
            else:
                results = response
            temp_start = temp_end
            data_points = (end - temp_start) / step

        if data_points > 0:
            response = self.query_range_limit(query, temp_start, end)
            if results:
                results["values"] = results["values"] + response["values"]
            else:
                _LOGGER.debug("Prometheus query has returned empty results.")
                results = response

        return results

    def query_range_limit(
        self, query: str, start: float, end: float, step: int = 30
    ) -> Optional[Dict]:
        data_points = (end - start) / step

        if data_points > 11000:
            _LOGGER.info("Limit query only supports 11,000 data points")
            return None

        results = None
        try:
            response = requests.get(
                self.PROMETHEUS_SERVER + "/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": f"{step}s"},
            )
            results = response.json()["data"]["result"][0]
        except Exception as ex:
            _LOGGER.exception("Prometheus error: %r", ex)
        return results

    def query(self, query: str) -> Optional[Dict]:
        results = []
        try:
            response = requests.get(
                self.PROMETHEUS_SERVER + "/api/v1/query", params={"query": query}
            )
            if response:
                results = response.json()["data"]["result"]
            else:
                _LOGGER.debug("Prometheus query has returned empty results.")
        except Exception as ex:
            _LOGGER.exception("error: %r", ex)

        return results


class PrometheusDataFetcher:
    def __init__(self, prometheus_server: str):
        self.prometheus = Prometheus(prometheus_server)

    @classmethod
    def clean_data(cls, df: pd.DataFrame, return_labels: list[str], limit=12):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(method="ffill", limit=limit)
        df = df.fillna(method="bfill", limit=limit)
        if df.columns[df.isna().any()].tolist():
            df.dropna(inplace=True)

        if df.empty:
            return pd.DataFrame()

        if "rollouts_pod_template_hash" in return_labels:
            df = df.reset_index()
            df = (
                pd.merge(
                    df, df[df.duplicated("timestamp", keep=False)], indicator=True, how="outer"
                )
                .query('_merge=="left_only"')
                .drop("_merge", axis=1)
            )
            df.set_index("timestamp", inplace=True)
            df.drop("rollouts_pod_template_hash", axis=1, inplace=True)
            df = df.sort_values(by=["timestamp"], ascending=True)
        return df

    def fetch_data(
        self,
        metric: str,
        labels: dict,
        hours: int = 36,
        scrape_interval: int = 30,
        return_labels=None,
    ) -> pd.DataFrame:
        _start_time = time.time()
        end_dt = datetime.now(pytz.utc)
        start_dt = end_dt - timedelta(hours=hours)

        df = self.prometheus.query_metric(
            metric_name=metric,
            labels_map=labels,
            return_labels=return_labels,
            start=start_dt.timestamp(),
            end=end_dt.timestamp(),
            step=scrape_interval,
        )

        try:
            df = self.clean_data(df, return_labels)
        except Exception as ex:
            _LOGGER.exception("Error while cleaning the data: Metric: %s, Error: %r", metric, ex)

        _LOGGER.info(
            "Time taken to fetch data: %s, for Metric: %s, for df shape: %s",
            time.time() - _start_time,
            metric,
            df.shape,
        )
        return df
