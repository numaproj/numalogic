import os
import unittest
from unittest.mock import patch, Mock

import numpy as np
from numalogic.registry import RedisRegistry
from orjson import orjson
from pynumaflow.function import Datum

from src._constants import TESTS_DIR
from src.entities import Status, StreamPayload, Header

# Make sure to import this in the end
from tests import redis_client, Preprocess
from tests.tools import get_prepoc_input, return_preproc_clf

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")
STREAM_NAN_DATA_PATH = os.path.join(DATA_DIR, "stream_nan.json")


class TestPreprocess(unittest.TestCase):
    preproc_input: Datum = None
    keys: list[str] = ["service-mesh", "1", "2"]

    @classmethod
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.preproc_input = get_prepoc_input(cls.keys, STREAM_DATA_PATH)

    def setUp(self) -> None:
        redis_client.flushall()

    @patch.object(RedisRegistry, "load", Mock(return_value=return_preproc_clf()))
    def test_preprocess(self):
        _out = Preprocess().run(self.keys, self.preproc_input)[0]
        payload = StreamPayload(**orjson.loads(_out.value))
        self.assertTrue(payload.data)
        self.assertTrue(payload.raw_data)
        self.assertIsInstance(payload, StreamPayload)
        self.assertEqual(payload.status, Status.PRE_PROCESSED)
        self.assertEqual(payload.header, Header.MODEL_INFERENCE)

    @patch.object(RedisRegistry, "load", Mock(return_value=None))
    def test_preprocess_no_clf(self):
        _out = Preprocess().run(self.keys, self.preproc_input)[0]
        payload = StreamPayload(**orjson.loads(_out.value))
        self.assertIsInstance(payload, StreamPayload)
        self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
        self.assertEqual(payload.header, Header.STATIC_INFERENCE)

    @patch.object(RedisRegistry, "load", Mock(return_value=return_preproc_clf()))
    def test_preprocess_with_nan(self):
        preproc_input = get_prepoc_input(self.keys, STREAM_NAN_DATA_PATH)
        _out = Preprocess().run(self.keys, preproc_input)[0]
        payload = StreamPayload(**orjson.loads(_out.value))

        df = payload.get_df()
        self.assertTrue(np.isfinite(df.values).all())
        self.assertTrue(payload.data)
        self.assertIsInstance(payload, StreamPayload)
        self.assertEqual(payload.status, Status.PRE_PROCESSED)
        self.assertEqual(payload.header, Header.MODEL_INFERENCE)


if __name__ == "__main__":
    unittest.main()
