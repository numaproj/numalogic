import logging

import numpy as np
import pandas as pd
import requests


LOGGER = logging.getLogger(__name__)


class Prometheus:
    def __init__(self, prometheus_server: str):
        self.url = prometheus_server

    def query_metric(
        self,
        metric_name: str,
        start: float,
        end: float,
        labels_map: dict = None,
        return_labels: list[str] = None,
        step: int = 30,
    ) -> pd.DataFrame:
        query = metric_name
        if labels_map:
            label_list = [str(key + "=" + "'" + labels_map[key] + "'") for key in labels_map]
            query = metric_name + "{" + ",".join(label_list) + "}"

        LOGGER.debug("Prometheus Query: %s", query)

        if end < start:
            raise ValueError("end_time must not be before start_time")

        results = self.query_range(query, start, end, step)

        frames = []
        for result in results:
            LOGGER.debug(
                "Prometheus query has returned %s values for %s.",
                len(result["values"]),
                result["metric"],
            )
            arr = np.ascontiguousarray(result["values"], dtype=np.float32)
            _df = pd.DataFrame(arr, columns=["timestamp", metric_name])

            data = result["metric"]
            if return_labels:
                for label in return_labels:
                    if label in data:
                        _df[label] = data[label]
            frames.append(_df)

        df = pd.DataFrame()
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.sort_values(by=["timestamp"], inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        return df

    def query_range(self, query: str, start: float, end: float, step: int = 30) -> list | None:
        results = []
        data_points = (end - start) / step
        temp_start = start
        while data_points > 11000:
            temp_end = temp_start + 11000 * step
            response = self.query_range_limit(query, temp_start, temp_end, step)
            for res in response:
                results.append(res)
            temp_start = temp_end
            data_points = (end - temp_start) / step

        if data_points > 0:
            response = self.query_range_limit(query, temp_start, end)
            for res in response:
                results.append(res)
        return results

    def query_range_limit(self, query: str, start: float, end: float, step: int = 30) -> list:
        results = []
        data_points = (end - start) / step

        if data_points > 11000:
            LOGGER.info("Limit query only supports 11,000 data points")
            return results
        try:
            response = requests.get(
                self.url + "/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": f"{step}s"},
            )
            results = response.json()["data"]["result"]
            LOGGER.debug(
                "Prometheus query has returned results for %s metric series.",
                len(results),
            )
        except Exception:
            LOGGER.exception("Prometheus error!")
            raise
        return results
