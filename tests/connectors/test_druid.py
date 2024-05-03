import datetime
import json
import logging
import os.path

import pandas as pd
import pydruid.query
import pytest
from deepdiff import DeepDiff
from freezegun import freeze_time
from pydruid.client import PyDruid
from pydruid.utils import aggregators
from pydruid.utils import postaggregator
from pydruid.utils.dimensions import DimensionSpec
from pydruid.utils.filters import Filter

from numalogic._constants import TESTS_DIR
from numalogic.connectors._config import Pivot
from numalogic.connectors.druid import (
    DruidFetcher,
    make_filter_pairs,
    build_params,
    postaggregator as _post_agg,
    aggregators as _agg,
)
from numalogic.tools.exceptions import DruidFetcherError

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def setup():
    end = datetime.datetime.now()
    start = end - datetime.timedelta(hours=36)
    fetcher = DruidFetcher(url="http://localhost:8888", endpoint="druid/v2/")
    return start.timestamp(), end.timestamp(), fetcher


@pytest.fixture
def mock_group_by(mocker):
    """Creates a Mock for PyDruid's groupby method."""

    def group_by(*_, **__):
        """Mock group by response from druid."""
        result = [
            {
                "version": "v1",
                "timestamp": "2023-07-11T01:36:00.000Z",
                "event": {"count": 5.0, "ciStatus": "success"},
            },
            {
                "version": "v1",
                "timestamp": "2023-07-11T01:37:00.000Z",
                "event": {"count": 1.0, "ciStatus": "success"},
            },
        ]
        query = pydruid.query.Query(query_dict={}, query_type="groupBy")
        query.parse(json.dumps(result))
        return query

    mocker.patch.object(PyDruid, "groupby", side_effect=group_by)


@pytest.fixture
def mock_group_by_doubles_sketch(mocker):
    """Creates a Mock for PyDruid's groupby method for doubles sketch."""

    def group_by(*_, **__):
        """Mock group by response for doubles sketch from druid."""
        result = [
            {
                "event": {
                    "agg0": 4,
                    "assetAlias": "identity.authn.signin",
                    "env": "prod",
                    "postAgg0": 21988,
                },
                "timestamp": "2023-09-06T07:50:00.000Z",
                "version": "v1",
            },
            {
                "event": {
                    "agg0": 22,
                    "assetAlias": "identity.authn.signin",
                    "env": "prod",
                    "postAgg0": 2237.7999997138977,
                },
                "timestamp": "2023-09-06T07:53:00.000Z",
                "version": "v1",
            },
        ]
        query = pydruid.query.Query(query_dict={}, query_type="groupBy")
        query.parse(json.dumps(result))
        return query

    mocker.patch.object(PyDruid, "groupby", side_effect=group_by)


@pytest.fixture
def mock_group_by_multi_column(mocker):
    """Creates a Mock for PyDruid's groupby method for doubles sketch."""

    def group_by(*_, **__):
        """Mock group by response for doubles sketch from druid."""
        result = [
            {
                "event": {
                    "service_alias": "identity.authn.signin",
                    "env": "prod",
                    "status": 200,
                    "http_status": "2xx",
                    "count": 20,
                },
                "timestamp": "2023-09-06T07:50:00.000Z",
                "version": "v1",
            },
            {
                "event": {
                    "service_alias": "identity.authn.signin",
                    "env": "prod",
                    "status": 500,
                    "http_status": "5xx",
                    "count": 10,
                },
                "timestamp": "2023-09-06T07:53:00.000Z",
                "version": "v1",
            },
        ]
        query = pydruid.query.Query(query_dict={}, query_type="groupBy")
        query.parse(json.dumps(result))
        return query

    mocker.patch.object(PyDruid, "groupby", side_effect=group_by)


def test_fetch(setup, mock_group_by):
    start, end, fetcher = setup
    _out = fetcher.fetch(
        filter_keys=["assetId"],
        filter_values=["5984175597303660107"],
        dimensions=["ciStatus"],
        datasource="tech-ip-customer-interaction-metrics",
        aggregations={"count": aggregators.doublesum("count")},
        group_by=["timestamp", "ciStatus"],
        hours=2,
        pivot=Pivot(
            index="timestamp",
            columns=["ciStatus"],
            value=["count"],
        ),
    )
    assert (2, 2) == _out.shape


def test_fetch_double_sketch(setup, mock_group_by_doubles_sketch):
    start, end, druid = setup
    _out = druid.fetch(
        filter_keys=["assetAlias"],
        filter_values=["accounting.core.qbowebapp"],
        dimensions=["assetAlias", "env"],
        datasource="coredevx-rum-perf-metrics",
        aggregations={"agg0": _agg.quantiles_doubles_sketch("valuesDoublesSketch", "agg0", 256)},
        post_aggregations={
            "postAgg0": _post_agg.QuantilesDoublesSketchToQuantile(
                output_name="agg0", field=postaggregator.Field("agg0"), fraction=0.9
            )
        },
        hours=2,
    )
    assert (2, 5) == _out.shape


def test_build_param(setup):
    start, end, druid = setup
    filter_pairs = make_filter_pairs(["ciStatus"], ["false"])
    expected = {
        "datasource": "foo",
        "dimensions": map(lambda d: DimensionSpec(dimension=d, output_name=d), ["bar"]),
        "filter": Filter(
            type="and",
            fields=[Filter(type="selector", dimension="ciStatus", value="false")],
        ),
        "granularity": "all",
        "aggregations": {},
        "post_aggregations": {},
        "intervals": "",
        "context": {
            "timeout": 10000,
            "configIds": list(filter_pairs),
            "source": "numalogic",
        },
    }

    actual = build_params(
        datasource="foo",
        dimensions=["bar"],
        filter_pairs=filter_pairs,
        granularity="all",
        hours=24.0,
        delay=3,
    )
    actual["intervals"] = ""
    diff = DeepDiff(expected, actual, ignore_order=True)
    assert {} == diff


def test_fetch_exception(mocker, setup):
    start, end, druid = setup
    mocker.patch.object(PyDruid, "groupby", side_effect=OSError)
    with pytest.raises(DruidFetcherError):
        _out = druid.fetch(
            filter_keys=["assetId"],
            filter_values=["5984175597303660107"],
            dimensions=["ciStatus"],
            datasource="customer-interaction-metrics",
            aggregations={"count": aggregators.doublesum("count")},
            group_by=["timestamp", "ciStatus"],
            hours=36,
            pivot=Pivot(
                index="timestamp",
                columns=["ciStatus"],
                value=["count"],
            ),
        )


@pytest.fixture()
def get_args():
    return {
        "filter_keys": ["assetId"],
        "filter_values": ["5984175597303660107"],
        "dimensions": ["ciStatus"],
        "datasource": "customer-interaction-metrics",
        "aggregations": {"count": aggregators.doublesum("count")},
        "group_by": ["timestamp", "ciStatus"],
        "pivot": Pivot(
            index="timestamp",
            columns=["ciStatus"],
            value=["count"],
        ),
        "hours": 24,
    }


@pytest.fixture()
def mock_query_fetch(mocker):
    def _fetch(*_, **params):
        interval = params["intervals"][0]
        start, end = interval.split("/")
        raw_df = pd.read_csv(
            os.path.join(TESTS_DIR, "resources", "data", "raw_druid.csv"), index_col=0
        )
        return raw_df[raw_df["timestamp"].between(start, end)]

    mocker.patch.object(DruidFetcher, "_fetch", side_effect=_fetch)


@freeze_time("2024-02-22 15:30:00")
def test_chunked_fetch(get_args, mock_query_fetch):
    fetcher = DruidFetcher(url="http://localhost:8888", endpoint="druid/v2/")
    chunked_out = fetcher.chunked_fetch(
        **get_args,
        chunked_hours=1,
    )
    full_out = fetcher.fetch(**get_args)
    assert not chunked_out.empty
    assert not full_out.empty
    assert chunked_out.shape == full_out.shape
    assert chunked_out.equals(full_out)


@freeze_time("2025-02-22 15:30:00")
def test_chunked_fetch_empty(get_args, mock_query_fetch):
    fetcher = DruidFetcher(url="http://localhost:8888", endpoint="druid/v2/")
    chunked_out = fetcher.chunked_fetch(
        **get_args,
        chunked_hours=1,
    )
    full_out = fetcher.fetch(**get_args)
    assert chunked_out.empty
    assert full_out.empty


def test_chunked_fetch_err(get_args):
    fetcher = DruidFetcher(url="http://localhost:8888", endpoint="druid/v2/")
    with pytest.raises(ValueError):
        fetcher.chunked_fetch(
            **get_args,
            chunked_hours=0,
        )


def test_multi_column_pivot(setup, mock_group_by_multi_column):
    start, end, fetcher = setup
    _out = fetcher.fetch(
        filter_keys=["service_alias"],
        filter_values=["identity.authn.signin"],
        dimensions=["http_status", "status"],
        datasource="ip-apigw-telegraf-druid",
        aggregations={"count": aggregators.doublesum("count")},
        group_by=["timestamp", "http_status", "status"],
        hours=2,
        pivot=Pivot(
            index="timestamp",
            columns=["http_status", "status"],
            value=["count"],
        ),
    )
    print(_out)
    assert (2, 5) == _out.shape
