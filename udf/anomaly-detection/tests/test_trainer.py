import json
import os
import unittest
from datetime import datetime
from typing import Union
from unittest.mock import patch, Mock

from pynumaflow.sink import Datum

from src._constants import TESTS_DIR
from src.connectors.prometheus import Prometheus
from tests.tools import (
    mock_argocd_query_metric,
    mock_rollout_query_metric,
    mock_rollout_query_metric2,
)
from tests import redis_client, Train

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


def as_datum(data: Union[str, bytes, dict], msg_id="1") -> Datum:
    if type(data) is not bytes:
        data = json.dumps(data).encode("utf-8")
    elif type(data) == dict:
        data = json.dumps(data)

    return Datum(
        sink_msg_id=msg_id, value=data, event_time=datetime.now(), watermark=datetime.now(), keys=[]
    )


class TestTrainer(unittest.TestCase):
    train_payload = {
        "uuid": "123124543",
        "composite_keys": {
            "namespace": "sandbox_numalogic_demo",
            "name": "metric_1",
            "hash_id": "123456789",
        },
    }

    train_payload2 = {
        "uuid": "123124543",
        "composite_keys": {
            "namespace": "sandbox_numalogic_demo",
            "name": "metric_1",
            "rollouts_pod_template_hash": "123456789",
            "app": "demo",
        },
    }

    def setUp(self) -> None:
        redis_client.flushall()

    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_argocd_query_metric()))
    def test_argocd_trainer_01(self):
        datums = [as_datum(self.train_payload)]
        _out = train(datums)
        self.assertTrue(_out.items()[0].success)
        self.assertEqual("1", _out.items()[0].id)
        self.assertEqual(1, len(_out.items()))

    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_argocd_query_metric()))
    def test_argocd_trainer_02(self):
        datums = [as_datum(self.train_payload), as_datum(self.train_payload, msg_id="2")]
        _out = train(datums)
        self.assertTrue(_out.items()[0].success)
        self.assertEqual("1", _out.items()[0].id)

        self.assertTrue(_out.items()[1].success)
        self.assertEqual("2", _out.items()[1].id)
        self.assertEqual(2, len(_out.items()))

    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_rollout_query_metric()))
    def test_argo_rollout_trainer_01(self):
        _out = train_rollout([as_datum(self.train_payload)])
        self.assertTrue(_out.items()[0].success)
        self.assertEqual("1", _out.items()[0].id)

    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_rollout_query_metric()))
    def test_argo_rollout_trainer_02(self):
        datums = [as_datum(self.train_payload), as_datum(self.train_payload, msg_id="2")]
        _out = train_rollout(datums)
        self.assertTrue(_out.items()[0].success)
        self.assertEqual("1", _out.items()[0].id)

        self.assertTrue(_out.items()[1].success)
        self.assertEqual("2", _out.items()[1].id)
        self.assertEqual(2, len(_out.items()))

    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_rollout_query_metric2()))
    def test_argo_rollout_trainer_03(self):
        _out = train_rollout([as_datum(self.train_payload2)])
        self.assertTrue(_out.items()[0].success)
        self.assertEqual("1", _out.items()[0].id)


if __name__ == "__main__":
    unittest.main()
