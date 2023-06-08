import os
import unittest

from orjson import orjson
from freezegun import freeze_time
from unittest.mock import patch, Mock

from numalogic.registry import RedisRegistry

from src._constants import TESTS_DIR, TRAIN_VTX_KEY
from src.entities import Status, StreamPayload, TrainerPayload, Header
from tests import redis_client, Threshold
from tests.tools import get_threshold_input, return_threshold_clf


DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestThreshold(unittest.TestCase):
    keys: list[str] = ["service-mesh", "1", "2"]

    @classmethod
    def setUpClass(cls) -> None:
        redis_client.flushall()

    def setUp(self) -> None:
        redis_client.flushall()

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(RedisRegistry, "load", Mock(return_value=return_threshold_clf()))
    def test_threshold(self):
        threshold_input = get_threshold_input(self.keys, STREAM_DATA_PATH)
        _out = Threshold().run(self.keys, threshold_input)[0]
        payload = StreamPayload(**orjson.loads(_out.value))
        self.assertTrue(payload.data)
        self.assertTrue(payload.raw_data)
        self.assertIsInstance(payload, StreamPayload)
        for metric in payload.metrics:
            self.assertEqual(payload.status[metric], Status.THRESHOLD)
            self.assertEqual(payload.header[metric], Header.MODEL_INFERENCE)
            self.assertGreater(payload.metadata[metric]["model_version"], 0)

    @patch.object(RedisRegistry, "load", Mock(return_value=return_threshold_clf()))
    def test_threshold_prev_stale_model(self):
        threshold_input = get_threshold_input(self.keys, STREAM_DATA_PATH, prev_model_stale=True)
        _out = Threshold().run(self.keys, threshold_input)
        for msg in _out:
            if TRAIN_VTX_KEY in msg.tags:
                train_payload = TrainerPayload(**orjson.loads(msg.value.decode("utf-8")))
                self.assertIsInstance(train_payload, TrainerPayload)
            else:
                payload = StreamPayload(**orjson.loads(msg.value.decode("utf-8")))
                self.assertIsInstance(payload, StreamPayload)
                for metric in payload.metrics:
                    self.assertEqual(payload.header[metric], Header.MODEL_STALE)
                    self.assertEqual(payload.status[metric], Status.THRESHOLD)
                    self.assertGreater(payload.metadata[metric]["model_version"], 0)

    @patch.object(RedisRegistry, "load", Mock(return_value=None))
    def test_threshold_no_prev_clf(self):
        threshold_input = get_threshold_input(self.keys, STREAM_DATA_PATH, prev_clf_exists=False)
        _out = Threshold().run(self.keys, threshold_input)
        for msg in _out:
            if TRAIN_VTX_KEY in msg.tags:
                train_payload = TrainerPayload(**orjson.loads(msg.value.decode("utf-8")))
                self.assertIsInstance(train_payload, TrainerPayload)
            else:
                payload = StreamPayload(**orjson.loads(msg.value.decode("utf-8")))
                self.assertIsInstance(payload, StreamPayload)
                for metric in payload.metrics:
                    self.assertEqual(payload.header[metric], Header.STATIC_INFERENCE)
                    self.assertEqual(payload.status[metric], Status.ARTIFACT_NOT_FOUND)
                    self.assertEqual(payload.metadata[metric]["model_version"], -1)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(RedisRegistry, "load", Mock(return_value=None))
    def test_threshold_no_clf(self):
        threshold_input = get_threshold_input(self.keys, STREAM_DATA_PATH)
        _out = Threshold().run(self.keys, threshold_input)
        for msg in _out:
            if TRAIN_VTX_KEY in msg.tags:
                train_payload = TrainerPayload(**orjson.loads(msg.value.decode("utf-8")))
                self.assertIsInstance(train_payload, TrainerPayload)
            else:
                payload = StreamPayload(**orjson.loads(msg.value.decode("utf-8")))
                self.assertIsInstance(payload, StreamPayload)
                for metric in payload.metrics:
                    self.assertEqual(payload.header[metric], Header.STATIC_INFERENCE)
                    self.assertEqual(payload.status[metric], Status.ARTIFACT_NOT_FOUND)
                    self.assertEqual(payload.metadata[metric]["model_version"], -1)


if __name__ == "__main__":
    unittest.main()
