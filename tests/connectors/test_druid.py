import json
import unittest
import datetime
from unittest.mock import patch, Mock
import pydruid.query
from pydruid.client import PyDruid
from pydruid.utils.aggregators import doublesum

from numalogic.connectors._config import Pivot
from numalogic.connectors.druid import DruidFetcher


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
    def test_fetch_data(self):
        _out = self.druid.fetch(
            filter_keys=["assetId"],
            filter_values=["5984175597303660107"],
            dimensions=["ciStatus"],
            datasource="customer-interaction-metrics",
            aggregations={"count": doublesum("count")},
            group_by=["timestamp", "ciStatus"],
            hours=36,
            pivot=Pivot(
                index="timestamp",
                columns=["ciStatus"],
                value=["count"],
            ),
        )
        self.assertEqual(_out.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
