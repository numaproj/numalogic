from copy import copy
from datetime import datetime, timedelta

import pytest
from wavefront_api_client import QueryResult

from numalogic.connectors import WavefrontFetcher
from numalogic.tools.exceptions import WavefrontFetcherError

DUMMY_URL = "https://dummy.wavefront.com"
DUMMY_TOKEN = "dummy_token"
DUMMY_OUT = QueryResult(
    **{
        "dimensions": None,
        "error_message": None,
        "error_type": None,
        "events": None,
        "granularity": 60,
        "name": "ts(iks.namespace.kube.hpa.status.desired.replicas, "
        "cluster='fdp-prd-usw2-k8s' and "
        "namespace='fdp-documentservice-usw2-prd') - "
        "ts(iks.namespace.app.pod.count, cluster='fdp-prd-usw2-k8s' and "
        "namespace='fdp-documentservice-usw2-prd')",
        "query": "ts(iks.namespace.kube.hpa.status.desired.replicas, "
        "cluster='fdp-prd-usw2-k8s' and "
        "namespace='fdp-documentservice-usw2-prd') - "
        "ts(iks.namespace.app.pod.count, cluster='fdp-prd-usw2-k8s' and "
        "namespace='fdp-documentservice-usw2-prd')",
        "spans": None,
        "stats": {
            "buffer_keys": 72,
            "cached_compacted_keys": None,
            "compacted_keys": 3,
            "compacted_points": 357,
            "cpu_ns": 398618692,
            "distributions": 0,
            "edges": 0,
            "hosts_used": None,
            "keys": 73,
            "latency": 413,
            "metrics": 427,
            "metrics_used": None,
            "points": 427,
            "queries": 17,
            "query_tasks": 0,
            "s3_keys": 0,
            "skipped_compacted_keys": 4,
            "spans": 0,
            "summaries": 427,
        },
        "timeseries": [
            {
                "data": [
                    [1726533000.0, 0.0],
                    [1726533060.0, 0.0],
                    [1726533120.0, 0.0],
                    [1726533180.0, 0.0],
                    [1726533240.0, 0.0],
                    [1726533300.0, 0.0],
                    [1726533360.0, 0.0],
                    [1726533420.0, 0.0],
                    [1726533480.0, 0.0],
                    [1726533540.0, 0.0],
                    [1726533600.0, 0.0],
                    [1726533660.0, 0.0],
                    [1726533720.0, 0.0],
                    [1726533780.0, 0.0],
                    [1726533840.0, 0.0],
                    [1726533900.0, 0.0],
                    [1726533960.0, 0.0],
                    [1726534020.0, 0.0],
                ],
                "host": "10.176.157.157:8080",
                "label": "iks.namespace.kube.hpa.status.desired.replicas",
                "tags": {
                    "assetId": "4615081310646958673",
                    "bu": "ip",
                    "cluster": "fdp-prd-usw2-k8s",
                    "container": "kube-state-metrics",
                    "endpoint": "http-metrics",
                    "env": "prod",
                    "horizontalpodautoscaler": "document-service-rollout-hpa",
                    "job": "kube-state-metrics-v2",
                    "namespace": "fdp-documentservice-usw2-prd",
                    "pod": "kube-state-metrics-v2-fc68fc5fb-kjzdc",
                    "prometheus": "addon-metricset-ns/k8s-prometheus",
                    "prometheus.replica": "prometheus-k8s-prometheus-0",
                    "service": "kube-state-metrics-v2",
                },
            }
        ],
        "trace_dimensions": [],
        "traces": None,
        "warnings": None,
    }
)

DUMMY_OUT_ERR = copy(DUMMY_OUT)
DUMMY_OUT_ERR.error_type = "QuerySyntaxError"
DUMMY_OUT_ERR.error_message = "Invalid query"

DUMMY_OUT_NO_TS = copy(DUMMY_OUT)
DUMMY_OUT_NO_TS.timeseries = None


@pytest.fixture
def wavefront_fetcher():
    return WavefrontFetcher(
        url=DUMMY_URL,
        api_token=DUMMY_TOKEN,
    )


def test_init():
    with pytest.raises(ValueError):
        WavefrontFetcher(url=DUMMY_URL)


def test_fetch_01(wavefront_fetcher, mocker):
    mocker.patch.object(wavefront_fetcher, "_call_api", return_value=DUMMY_OUT)

    df = wavefront_fetcher.fetch(
        metric="iks.namespace.kube.hpa.status.desired.replicas",
        start=datetime.now() - timedelta(days=1),
        filters={"cluster": "fdp-prd-usw2-k8s", "namespace": "fdp-documentservice-usw2-prd"},
        end=datetime.now(),
    )
    assert df.shape == (18, 1)
    assert df.columns == ["value"]
    assert df.index.name == "timestamp"


def test_fetch_02(wavefront_fetcher, mocker):
    mocker.patch.object(wavefront_fetcher, "_call_api", return_value=DUMMY_OUT)

    df = wavefront_fetcher.fetch(
        metric="iks.namespace.kube.hpa.status.desired.replicas",
        start=datetime.now() - timedelta(days=1),
        end=datetime.now(),
    )
    assert df.shape == (18, 1)
    assert df.columns == ["value"]
    assert df.index.name == "timestamp"


def test_raw_fetch(wavefront_fetcher, mocker):
    mocker.patch.object(wavefront_fetcher, "_call_api", return_value=DUMMY_OUT)

    df = wavefront_fetcher.raw_fetch(
        query="ts(iks.namespace.kube.hpa.status.desired.replicas, cluster='{cluster}' and "
        "namespace='{namespace}') - ts(iks.namespace.app.pod.count, cluster='{cluster}' and "
        "namespace='{namespace}')",
        start=datetime.now() - timedelta(minutes=5),
        filters={"cluster": "fdp-prd-usw2-k8s", "namespace": "fdp-documentservice-usw2-prd"},
        end=datetime.now(),
    )
    assert df.shape == (18, 1)
    assert df.columns == ["value"]
    assert df.index.name == "timestamp"


def test_fetch_err_01(wavefront_fetcher, mocker):
    mocker.patch.object(wavefront_fetcher, "_call_api", return_value=DUMMY_OUT_ERR)

    with pytest.raises(WavefrontFetcherError):
        wavefront_fetcher.fetch(
            metric="some_metric",
            start=datetime.now() - timedelta(days=1),
            filters={"cluster": "some_cluster", "namespace": "some_ns"},
            end=datetime.now(),
        )


def test_fetch_err_02(wavefront_fetcher, mocker):

    mocker.patch.object(wavefront_fetcher, "_call_api", return_value=DUMMY_OUT_NO_TS)

    with pytest.raises(WavefrontFetcherError):
        wavefront_fetcher.fetch(
            metric="some_metric",
            start=datetime.now() - timedelta(days=1),
            filters={"cluster": "some_cluster", "namespace": "some_ns"},
            end=datetime.now(),
        )


def test_raw_fetch_err(wavefront_fetcher, mocker):
    mocker.patch.object(wavefront_fetcher, "_call_api", return_value=DUMMY_OUT)

    with pytest.raises(WavefrontFetcherError):
        wavefront_fetcher.raw_fetch(
            query="ts(iks.namespace.kube.hpa.status.desired.replicas, cluster='{cluster}' and "
            "namespace='{namespace}') - ts(iks.namespace.app.pod.count, cluster='{cluster}' and "
            "namespace='{namespace}')",
            start=datetime.now() - timedelta(minutes=5),
            filters={"randomkey": "fdp-prd-usw2-k8s", "namespace": "fdp-documentservice-usw2-prd"},
            end=datetime.now(),
        )
