import os
import unittest

from orjson import orjson
from unittest.mock import patch, Mock
from freezegun import freeze_time

from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import RedisRegistry
from pynumaflow.function import Datum

from src._constants import TESTS_DIR
from src.entities import Status, StreamPayload, Header
from tests import redis_client, Inference
from tests.tools import (
    get_inference_input,
    return_stale_model,
    return_mock_lstmae,
)

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestInference(unittest.TestCase):
    inference_input: Datum = None
    keys: list[str] = ["service-mesh", "1", "2"]

    @classmethod
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.inference_input = get_inference_input(cls.keys, STREAM_DATA_PATH)

    def setUp(self) -> None:
        redis_client.flushall()

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(RedisRegistry, "load", Mock(return_value=return_mock_lstmae()))
    def test_inference(self):
        _out = Inference().run(self.keys, self.inference_input)
        msg = _out.items()[0]
        payload = StreamPayload(**orjson.loads(msg.value))
        self.assertTrue(payload.data)
        self.assertTrue(payload.raw_data)
        self.assertIsInstance(payload, StreamPayload)
        for metric in payload.metrics:
                self.assertEqual(payload.status[metric], Status.INFERRED)
                self.assertEqual(payload.header[metric], Header.MODEL_INFERENCE)
                self.assertGreater(payload.metadata[metric]["model_version"], 0)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(RedisRegistry, "load", Mock(return_value=return_mock_lstmae()))
    @patch.object(AutoencoderTrainer, "predict", Mock(side_effect=RuntimeError))
    def test_inference_err(self):
        _out = Inference().run(self.keys, self.inference_input)
        msg = _out.items()[0]
        payload = StreamPayload(**orjson.loads(msg.value))
        self.assertTrue(payload.data)
        self.assertTrue(payload.raw_data)
        self.assertIsInstance(payload, StreamPayload)
        for metric in payload.metrics:
                self.assertEqual(payload.status[metric], Status.RUNTIME_ERROR)
                self.assertEqual(payload.header[metric], Header.STATIC_INFERENCE)
                self.assertEqual(payload.metadata[metric]["model_version"], -1)

    @patch.object(RedisRegistry, "load", Mock(return_value=None))
    def test_no_model(self):
        _out = Inference().run(self.keys, self.inference_input)
        msg = _out.items()[0]
        payload = StreamPayload(**orjson.loads(msg.value))
        self.assertTrue(payload.data)
        self.assertTrue(payload.raw_data)
        self.assertIsInstance(payload, StreamPayload)
        for metric in payload.metrics:
            self.assertEqual(payload.status[metric], Status.ARTIFACT_NOT_FOUND)
            self.assertEqual(payload.header[metric], Header.STATIC_INFERENCE)
            self.assertEqual(payload.metadata[metric]["model_version"], -1)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(RedisRegistry, "load", Mock(return_value=return_mock_lstmae()))
    def test_no_prev_model(self):
        inference_input = get_inference_input(self.keys, STREAM_DATA_PATH, prev_clf_exists=False)
        _out = Inference().run(self.keys, inference_input)
        msg = _out.items()[0]
        payload = StreamPayload(**orjson.loads(msg.value))
        self.assertTrue(payload.data)
        self.assertTrue(payload.raw_data)
        self.assertIsInstance(payload, StreamPayload)
        for metric in payload.metrics:
            self.assertEqual(payload.status[metric], Status.ARTIFACT_NOT_FOUND)
            self.assertEqual(payload.header[metric], Header.STATIC_INFERENCE)
            self.assertEqual(payload.metadata[metric]["model_version"], -1)

    @patch.object(RedisRegistry, "load", Mock(return_value=return_stale_model()))
    def test_stale_model(self):
        _out = Inference().run(self.keys, self.inference_input)
        msg = _out.items()[0]
        payload = StreamPayload(**orjson.loads(msg.value))
        self.assertTrue(payload.data)
        self.assertTrue(payload.raw_data)
        self.assertIsInstance(payload, StreamPayload)
        for metric in payload.metrics:
            self.assertEqual(payload.status[metric], Status.INFERRED)
            self.assertEqual(payload.header[metric], Header.MODEL_STALE)
            self.assertGreater(payload.metadata[metric]["model_version"], 0)


if __name__ == "__main__":
    unittest.main()
