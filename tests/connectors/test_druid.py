import json
import logging
import unittest
import datetime
from unittest.mock import patch, Mock

import pydruid.query
from pydruid.client import PyDruid
from pydruid.utils.dimensions import DimensionSpec
from pydruid.utils import aggregators
from pydruid.utils import postaggregator
from pydruid.utils.filters import Filter
from deepdiff import DeepDiff

from numalogic.connectors._config import Pivot
from numalogic.connectors.druid import (
    DruidFetcher,
    make_filter_pairs,
    build_params,
    postaggregator as _post_agg,
    aggregators as _agg,
)

logging.basicConfig(level=logging.DEBUG)


def mock_group_by(*_, **__):
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


def mock_group_by_doubles_sketch(*_, **__):
    """Mock group by response for doubles sketch from druid."""
    result = [
        {
            "event": {
                "agg0": 4,
                "assetAlias": "Intuit.identity.authn.signin",
                "env": "prod",
                "postAgg0": 21988,
            },
            "timestamp": "2023-09-06T07:50:00.000Z",
            "version": "v1",
        },
        {
            "event": {
                "agg0": 22,
                "assetAlias": "Intuit.identity.authn.signin",
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


class TestDruid(unittest.TestCase):
    start = None
    end = None
    prom = None

    @classmethod
    def setUpClass(cls) -> None:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(hours=36)
        cls.start = start.timestamp()
        cls.end = end.timestamp()
        cls.druid = DruidFetcher(url="http://localhost:8888", endpoint="druid/v2/")

    @patch.object(PyDruid, "groupby", Mock(return_value=mock_group_by()))
    def test_fetch(self):
        _out = self.druid.fetch(
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
        self.assertEqual((2, 2), _out.shape)

    @patch.object(PyDruid, "groupby", Mock(return_value=mock_group_by_doubles_sketch()))
    def test_fetch_double_sketch(self):
        _out = self.druid.fetch(
            filter_keys=["assetAlias"],
            filter_values=["Intuit.accounting.core.qbowebapp"],
            dimensions=["assetAlias", "env"],
            datasource="coredevx-rum-perf-metrics",
            aggregations={
                "agg0": _agg.quantiles_doubles_sketch("valuesDoublesSketch", "agg0", 256)
            },
            post_aggregations={
                "postAgg0": _post_agg.QuantilesDoublesSketchToQuantile(
                    output_name="agg0", field=postaggregator.Field("agg0"), fraction=0.9
                )
            },
            hours=2,
        )
        self.assertEqual((2, 5), _out.shape)

    def test_build_param(self):
        expected = {
            "datasource": "foo",
            "dimensions": map(lambda d: DimensionSpec(dimension=d, output_name=d), ["bar"]),
            "filter": [Filter(type="selector", dimension="ciStatus", value="false")],
            "granularity": "all",
            "aggregations": "",
            "intervals": "",
        }

        filter_pairs = make_filter_pairs(["ciStatus"], ["false"])
        actual = build_params(
            datasource="foo",
            dimensions=["bar"],
            filter_pairs=filter_pairs,
            granularity="all",
            hours=24.0,
            delay=3,
        )
        diff = DeepDiff(expected, actual).get("values_changed", {})
        self.assertDictEqual({}, diff)

    @patch.object(PyDruid, "groupby", Mock(side_effect=OSError))
    def test_fetch_exception(self):
        _out = self.druid.fetch(
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
        self.assertTrue(_out.empty)


if __name__ == "__main__":
    unittest.main()
