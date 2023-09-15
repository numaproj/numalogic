import logging
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock

from orjson import orjson
from requests import Response

from numalogic.connectors import PrometheusFetcher
from numalogic.tools.exceptions import PrometheusFetcherError, PrometheusInvalidResponseError

logging.basicConfig(level=logging.DEBUG)


def _mock_response():
    response = MagicMock()
    response.status_code = 200
    response.text = orjson.dumps(
        {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {
                            "__name__": "namespace_asset_pod_cpu_utilization",
                            "numalogic": "true",
                            "namespace": "sandbox-numalogic-demo",
                        },
                        "values": [
                            [1656334767.73, "14.744611739611193"],
                            [1656334797.73, "14.73040822323633"],
                        ],
                    }
                ],
            },
        }
    )
    return response


def _mock_query_range():
    return [
        {
            "metric": {
                "__name__": "namespace_asset_pod_cpu_utilization",
                "assetAlias": "sandbox.numalogic.demo",
                "numalogic": "true",
                "namespace": "sandbox-numalogic-demo",
            },
            "values": [[1656334767.73, "14.744611739611193"], [1656334797.73, "14.73040822323633"]],
        }
    ]


def _mock_w_return_labels():
    return [
        {
            "metric": {
                "__name__": "namespace_app_rollouts_http_request_error_rate",
                "assetAlias": "sandbox.numalogic.demo",
                "numalogic": "true",
                "namespace": "sandbox-numalogic-demo",
                "rollouts_pod_template_hash": "7b4b4f9f9d",
            },
            "values": [[1656334767.73, "10.0"], [1656334797.73, "12.0"]],
        },
        {
            "metric": {
                "__name__": "namespace_app_rollouts_http_request_error_rate",
                "assetAlias": "sandbox.numalogic.demo",
                "numalogic": "true",
                "namespace": "sandbox-numalogic-demo",
                "rollouts_pod_template_hash": "5b4b4f9f9d",
            },
            "values": [[1656334767.73, "11.0"], [1656334797.73, "13.0"]],
        },
    ]


class TestPrometheusFetcher(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = PrometheusFetcher(prometheus_server="http://localhost:9090")

    @patch("requests.get", Mock(return_value=_mock_response()))
    def test_fetch(self):
        df = self.fetcher.fetch(
            metric_name="namespace_asset_pod_cpu_utilization",
            start=datetime.now() - timedelta(hours=1),
            filters={"namespace": "sandbox-numalogic-demo"},
        )
        self.assertEqual(df.shape, (2, 2))
        self.assertListEqual(
            df.columns.to_list(), ["timestamp", "namespace_asset_pod_cpu_utilization"]
        )

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_w_return_labels()))
    def test_fetch_return_labels(self):
        metric = "namespace_app_rollouts_http_request_error_rate"
        df = self.fetcher.fetch(
            metric_name=metric,
            start=datetime.now() - timedelta(hours=1),
            filters={"namespace": "sandbox-numalogic-demo"},
            return_labels=["rollouts_pod_template_hash"],
            aggregate=True,
        )
        self.assertEqual(df.shape, (2, 2))
        self.assertListEqual(df.columns.to_list(), ["timestamp", metric])
        self.assertListEqual([10.5, 12.5], df[metric].to_list())

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=[]))
    def test_fetch_no_data(self):
        df = self.fetcher.fetch(
            metric_name="namespace_asset_pod_cpu_utilization",
            start=datetime.now() - timedelta(hours=1),
            filters={"namespace": "sandbox-numalogic-demo"},
        )
        self.assertTrue(df.empty)

    @patch("requests.get", Mock(side_effect=Exception("Test exception")))
    def test_fetch_url_err(self):
        with self.assertRaises(PrometheusFetcherError):
            self.fetcher.fetch(
                metric_name="namespace_asset_pod_cpu_utilization",
                start=datetime.now() - timedelta(hours=1),
                filters={"namespace": "sandbox-numalogic-demo"},
            )

    @patch("requests.get", Mock(return_value=Response()))
    def test_fetch_response_err(self):
        with self.assertRaises(PrometheusInvalidResponseError):
            self.fetcher.fetch(
                metric_name="namespace_asset_pod_cpu_utilization",
                start=datetime.now() - timedelta(hours=1),
                filters={"namespace": "sandbox-numalogic-demo"},
            )

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_query_range()))
    def test_fetch_raw(self):
        df = self.fetcher.raw_fetch(
            query='namespace_asset_pod_cpu_utilization{namespace="sandbox-numalogic-demo"}',
            start=datetime.now() - timedelta(hours=1),
        )
        self.assertEqual(df.shape, (2, 2))
        self.assertListEqual(
            df.columns.to_list(), ["timestamp", "namespace_asset_pod_cpu_utilization"]
        )

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_w_return_labels()))
    def test_fetch_raw_return_labels(self):
        metric = "namespace_app_rollouts_http_request_error_rate"
        df = self.fetcher.raw_fetch(
            query="namespace_app_rollouts_http_request_error_rate{namespace='sandbox-numalogic-demo'}",
            start=datetime.now() - timedelta(hours=1),
            return_labels=["rollouts_pod_template_hash"],
            aggregate=True,
        )
        self.assertEqual(df.shape, (2, 2))
        self.assertListEqual(df.columns.to_list(), ["timestamp", metric])
        self.assertListEqual([10.5, 12.5], df[metric].to_list())

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=[]))
    def test_fetch_raw_no_data(self):
        df = self.fetcher.raw_fetch(
            query='namespace_asset_pod_cpu_utilization{namespace="sandbox-numalogic-demo"}',
            start=datetime.now() - timedelta(hours=1),
        )
        self.assertTrue(df.empty)

    def test_start_end_err(self):
        with self.assertRaises(ValueError):
            self.fetcher.fetch(
                metric_name="namespace_asset_pod_cpu_utilization",
                start=datetime.now() - timedelta(hours=1),
                end=datetime.now() - timedelta(hours=2),
            )


if __name__ == "__main__":
    unittest.main()
