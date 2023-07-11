import requests
import datetime
import unittest
from unittest.mock import patch, Mock, MagicMock

from src.connectors.prometheus import Prometheus


def mock_multiple_metrics(*_, **__):
    result = [
        {
            "metric": {
                "__name__": "namespace_app_rollouts_http_request_error_rate",
                "assetAlias": "sandbox.numalogic.demo",
                "numalogic": "true",
                "namespace": "sandbox-numalogic-demo",
                "rollouts_pod_template_hash": "7b4b4f9f9d",
            },
            "values": [[1656334767.73, "14.744611739611193"], [1656334797.73, "14.73040822323633"]],
        },
        {
            "metric": {
                "__name__": "namespace_app_rollouts_http_request_error_rate",
                "assetAlias": "sandbox.numalogic.demo",
                "numalogic": "true",
                "namespace": "sandbox-numalogic-demo",
                "rollouts_pod_template_hash": "5b4b4f9f9d",
            },
            "values": [[1656334767.73, "14.744611739611193"], [1656334797.73, "14.73040822323633"]],
        },
    ]

    return result


def mock_query_range(*_, **__):
    result = [
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

    return result


def mock_response(*_, **__):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
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
    return response


class TestPrometheus(unittest.TestCase):
    start = None
    end = None
    prom = None

    @classmethod
    def setUpClass(cls) -> None:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(hours=36)
        cls.start = start.timestamp()
        cls.end = end.timestamp()
        cls.prom = Prometheus(prometheus_server="http://localhost:8490")

    @patch.object(Prometheus, "query_range", Mock(return_value=mock_query_range()))
    def test_query_metric1(self):
        _out = self.prom.query_metric(
            metric_name="namespace_app_pod_http_server_requests_errors",
            start=self.start,
            end=self.end,
        )
        self.assertEqual(_out.shape, (2, 2))

    @patch.object(Prometheus, "query_range", Mock(return_value=mock_query_range()))
    def test_query_metric2(self):
        _out = self.prom.query_metric(
            metric_name="namespace_app_pod_http_server_requests_errors",
            labels_map={"namespace": "sandbox-rollout-numalogic-demo"},
            start=self.start,
            end=self.end,
        )
        self.assertEqual(_out.shape, (2, 2))

    @patch.object(Prometheus, "query_range", Mock(return_value=mock_query_range()))
    def test_query_metric3(self):
        _out = self.prom.query_metric(
            metric_name="namespace_app_pod_http_server_requests_errors",
            labels_map={"namespace": "sandbox-numalogic-demo"},
            return_labels=["namespace"],
            start=self.start,
            end=self.end,
        )
        self.assertEqual(_out.shape, (2, 3))

    @patch.object(Prometheus, "query_range", Mock(return_value=mock_multiple_metrics()))
    def test_query_metric4(self):
        _out = self.prom.query_metric(
            metric_name="namespace_app_pod_http_server_requests_errors",
            labels_map={"namespace": "sandbox-numalogic-demo"},
            return_labels=["rollouts_pod_template_hash"],
            start=self.start,
            end=self.end,
        )
        self.assertEqual(_out.shape, (4, 3))
        self.assertEqual(_out["rollouts_pod_template_hash"].unique().shape[0], 2)

    @patch.object(requests, "get", Mock(return_value=mock_response()))
    def test_query_range(self):
        _out = self.prom.query_range(
            query="namespace_asset_pod_cpu_utilization{" "namespace='sandbox-numalogic-demo'}",
            start=self.start,
            end=self.end,
        )
        self.assertEqual(len(_out), 1)
        self.assertEqual(len(_out[0]["values"]), 2)

    @patch.object(requests, "get", Mock(return_value=mock_response()))
    def test_query(self):
        _out = self.prom.query(
            query="namespace_asset_pod_cpu_utilization{" "namespace='sandbox-numalogic-demo'}"
        )
        self.assertEqual(len(_out), 1)


if __name__ == "__main__":
    unittest.main()
