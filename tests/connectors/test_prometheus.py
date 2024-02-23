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


def _mock_mv():
    return [
        {
            "metric": {
                "__name__": "namespace_app_rollouts_cpu_utilization",
                "app": "odl-graphql",
                "assetId": "723971233226699519",
                "namespace": "odl-odlgraphql-usw2-e2e",
                "numalogic": "true",
                "prometheus": "addon-metricset-ns/k8s-prometheus",
                "role": "stable",
                "rollouts_pod_template_hash": "79cd8c7fb7",
            },
            "values": [
                [1700074605, "1.3404451281798757"],
                [1700074635, "1.58785690280504"],
                [1700074665, "2.0855656835916365"],
                [1700074695, "0.9484743464920472"],
                [1700074725, "1.2003565820676902"],
                [1700074755, "2.812905753652541"],
                [1700074785, "1.676252312582089"],
            ],
        },
        {
            "metric": {
                "__name__": "namespace_app_rollouts_cpu_utilization",
                "app": "odl-graphql",
                "assetId": "723971233226699519",
                "namespace": "odl-odlgraphql-usw2-e2e",
                "numalogic": "true",
                "prometheus": "addon-metricset-ns/k8s-prometheus",
                "role": "canary",
                "rollouts_pod_template_hash": "79cd8c7f00",
            },
            "values": [
                [1700074725, "5.2003565820676902"],
                [1700074755, "8.812905753652541"],
                [1700074785, "7.676252312582089"],
            ],
        },
        {
            "metric": {
                "__name__": "namespace_app_rollouts_http_request_error_rate",
                "aiops_argorollouts": "true",
                "app": "odl-graphql",
                "assetId": "723971233226699519",
                "namespace": "odl-odlgraphql-usw2-e2e",
                "numalogic": "true",
                "prometheus": "addon-metricset-ns/k8s-prometheus",
                "role": "stable",
                "rollouts_pod_template_hash": "79cd8c7fb7",
            },
            "values": [
                [1700074605, "0"],
                [1700074635, "0"],
                [1700074665, "0"],
                [1700074695, "0"],
                [1700074725, "0"],
                [1700074755, "0"],
                [1700074785, "0"],
            ],
        },
        {
            "metric": {
                "__name__": "namespace_app_rollouts_memory_utilization",
                "app": "odl-graphql",
                "assetId": "723971233226699519",
                "namespace": "odl-odlgraphql-usw2-e2e",
                "numalogic": "true",
                "prometheus": "addon-metricset-ns/k8s-prometheus",
                "role": "stable",
                "rollouts_pod_template_hash": "79cd8c7fb7",
            },
            "values": [
                [1700074605, "14.431392785274621"],
                [1700074635, "14.312004320549242"],
                [1700074665, "14.309229995265152"],
                [1700074695, "14.311449455492424"],
                [1700074725, "14.312096798058713"],
                [1700074755, "14.436201615767045"],
                [1700074785, "14.329112659801137"],
            ],
        },
    ]


def _mock_uv():
    return [
        {
            "metric": {
                "__name__": "namespace_app_rollouts_memory_utilization",
                "app": "odl-graphql",
                "assetId": "723971233226699519",
                "namespace": "odl-odlgraphql-usw2-e2e",
                "numalogic": "true",
                "prometheus": "addon-metricset-ns/k8s-prometheus",
                "role": "stable",
                "rollouts_pod_template_hash": "79cd8c7fb7",
            },
            "values": [
                [1700089988, "14.647327769886363"],
                [1700090018, "14.507409298058713"],
                [1700090048, "14.507779208096592"],
                [1700090078, "14.632253935842803"],
                [1700090108, "14.51129335345644"],
                [1700090138, "14.514437588778408"],
                [1700090168, "14.504357540246213"],
            ],
        }
    ]


def _mock_no_metric():
    return [
        {
            "metric": {
                "app": "odl-graphql",
                "assetId": "723971233226699519",
                "namespace": "odl-odlgraphql-usw2-e2e",
                "numalogic": "true",
                "prometheus": "addon-metricset-ns/k8s-prometheus",
                "role": "stable",
                "rollouts_pod_template_hash": "79cd8c7fb7",
            },
            "values": [
                [1700089988, "14.647327769886363"],
            ],
        }
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
        self.assertEqual(df.shape, (2, 1))
        self.assertListEqual(df.columns.to_list(), ["namespace_asset_pod_cpu_utilization"])
        self.assertEqual(df.index.name, "timestamp")

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
        self.assertEqual(df.shape, (2, 1))
        self.assertListEqual(df.columns.to_list(), [metric])
        self.assertEqual(df.index.name, "timestamp")
        self.assertListEqual([10.5, 12.5], df[metric].to_list())

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_mv()))
    def test_fetch_mv_01(self):
        df = self.fetcher.fetch(
            filters={"namespace": "odl-odlgraphql-usw2-e2e", "numalogic": "true"},
            start=datetime.now() - timedelta(minutes=3),
            aggregate=True,
        )
        self.assertEqual(df.shape, (7, 3))
        self.assertListEqual(
            df.columns.to_list(),
            [
                "namespace_app_rollouts_cpu_utilization",
                "namespace_app_rollouts_http_request_error_rate",
                "namespace_app_rollouts_memory_utilization",
            ],
        )
        self.assertEqual(df.index.name, "timestamp")

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_mv()))
    def test_fetch_mv_02(self):
        df = self.fetcher.fetch(
            filters={"namespace": "odl-odlgraphql-usw2-e2e", "numalogic": "true"},
            start=datetime.now() - timedelta(minutes=3),
            aggregate=False,
        )
        self.assertEqual(df.shape, (10, 3))
        self.assertListEqual(
            df.columns.to_list(),
            [
                "namespace_app_rollouts_cpu_utilization",
                "namespace_app_rollouts_http_request_error_rate",
                "namespace_app_rollouts_memory_utilization",
            ],
        )
        self.assertEqual(df.index.name, "timestamp")

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
        self.assertEqual(df.shape, (2, 1))
        self.assertListEqual(df.columns.to_list(), ["namespace_asset_pod_cpu_utilization"])
        self.assertEqual(df.index.name, "timestamp")

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_w_return_labels()))
    def test_fetch_raw_return_labels(self):
        metric = "namespace_app_rollouts_http_request_error_rate"
        df = self.fetcher.raw_fetch(
            query="namespace_app_rollouts_http_request_error_rate{namespace='sandbox-numalogic-demo'}",
            start=datetime.now() - timedelta(hours=1),
            return_labels=["rollouts_pod_template_hash"],
            aggregate=True,
        )
        self.assertTupleEqual((2, 1), df.shape)
        self.assertListEqual(df.columns.to_list(), [metric])
        self.assertEqual(df.index.name, "timestamp")
        self.assertListEqual([10.5, 12.5], df[metric].to_list())

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=[]))
    def test_fetch_raw_no_data(self):
        df = self.fetcher.raw_fetch(
            query='namespace_asset_pod_cpu_utilization{namespace="sandbox-numalogic-demo"}',
            start=datetime.now() - timedelta(hours=1),
        )
        self.assertTrue(df.empty)

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_mv()))
    def test_fetch_raw_mv_01(self):
        df = self.fetcher.raw_fetch(
            query='{namespace="odl-odlgraphql-usw2-e2e", numalogic="true"}',
            start=datetime.now() - timedelta(minutes=3),
            aggregate=True,
        )
        self.assertEqual(df.shape, (7, 3))
        self.assertListEqual(
            df.columns.to_list(),
            [
                "namespace_app_rollouts_cpu_utilization",
                "namespace_app_rollouts_http_request_error_rate",
                "namespace_app_rollouts_memory_utilization",
            ],
        )
        self.assertEqual(df.index.name, "timestamp")

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_mv()))
    def test_fetch_raw_mv_02(self):
        df = self.fetcher.raw_fetch(
            query='{namespace="odl-odlgraphql-usw2-e2e", numalogic="true"}',
            start=datetime.now() - timedelta(minutes=3),
            aggregate=False,
        )
        self.assertEqual(df.shape, (10, 3))
        self.assertListEqual(
            df.columns.to_list(),
            [
                "namespace_app_rollouts_cpu_utilization",
                "namespace_app_rollouts_http_request_error_rate",
                "namespace_app_rollouts_memory_utilization",
            ],
        )
        self.assertEqual(df.index.name, "timestamp")

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_mv()))
    def test_fetch_raw_mv_03(self):
        df = self.fetcher.raw_fetch(
            query='{namespace="odl-odlgraphql-usw2-e2e", numalogic="true"}',
            start=datetime.now() - timedelta(minutes=3),
            return_labels=["rollouts_pod_template_hash"],
            aggregate=True,
        )
        self.assertEqual(df.shape, (7, 3))
        self.assertListEqual(
            df.columns.to_list(),
            [
                "namespace_app_rollouts_cpu_utilization",
                "namespace_app_rollouts_http_request_error_rate",
                "namespace_app_rollouts_memory_utilization",
            ],
        )
        self.assertEqual(df.index.name, "timestamp")

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_mv()))
    def test_fetch_raw_mv_04(self):
        df = self.fetcher.raw_fetch(
            query='{namespace="odl-odlgraphql-usw2-e2e", numalogic="true"}',
            start=datetime.now() - timedelta(minutes=3),
            return_labels=["rollouts_pod_template_hash"],
            aggregate=False,
        )
        self.assertEqual(df.shape, (10, 4))
        self.assertListEqual(
            df.columns.to_list(),
            [
                "rollouts_pod_template_hash",
                "namespace_app_rollouts_cpu_utilization",
                "namespace_app_rollouts_http_request_error_rate",
                "namespace_app_rollouts_memory_utilization",
            ],
        )
        self.assertEqual(df.index.name, "timestamp")

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_mv()))
    def test_fetch_raw_mv_05(self):
        df = self.fetcher.raw_fetch(
            query="namespace_app_rollouts_memory_utilization"
            '{namespace="odl-odlgraphql-usw2-e2e", numalogic="true"}',
            start=datetime.now() - timedelta(minutes=3),
            aggregate=True,
        )
        self.assertEqual(df.shape, (7, 3))
        self.assertListEqual(
            df.columns.to_list(),
            [
                "namespace_app_rollouts_cpu_utilization",
                "namespace_app_rollouts_http_request_error_rate",
                "namespace_app_rollouts_memory_utilization",
            ],
        )
        self.assertEqual(df.index.name, "timestamp")

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_uv()))
    def test_fetch_raw_mv_06(self):
        df = self.fetcher.raw_fetch(
            query="namespace_app_rollouts_memory_utilization"
            '{namespace="odl-odlgraphql-usw2-e2e", numalogic="true"}',
            start=datetime.now() - timedelta(minutes=3),
            aggregate=True,
        )
        self.assertEqual(df.shape, (7, 1))
        self.assertListEqual(
            df.columns.to_list(),
            [
                "namespace_app_rollouts_memory_utilization",
            ],
        )
        self.assertEqual(df.index.name, "timestamp")

    def test_start_end_err(self):
        with self.assertRaises(ValueError):
            self.fetcher.fetch(
                metric_name="namespace_asset_pod_cpu_utilization",
                start=datetime.now() - timedelta(hours=1),
                end=datetime.now() - timedelta(hours=2),
            )

    @patch.object(PrometheusFetcher, "_api_query_range", Mock(return_value=_mock_no_metric()))
    def test_no_metric_name_err(self):
        with self.assertRaises(PrometheusInvalidResponseError):
            self.fetcher.raw_fetch(
                query="namespace_app_rollouts_memory_utilization"
                '{namespace="odl-odlgraphql-usw2-e2e", numalogic="true"}',
                start=datetime.now() - timedelta(minutes=3),
                aggregate=True,
            )

    def test_query_comparision(self):
        q1 = "{namespace='odl-odlgraphql-usw2-e2e',numalogic='true'}"
        q2 = self.fetcher.build_query(
            "", {"namespace": "odl-odlgraphql-usw2-e2e", "numalogic": "true"}
        )
        self.assertEqual(q1, q2)


if __name__ == "__main__":
    unittest.main()
