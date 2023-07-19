import os
import orjson
import unittest

from freezegun import freeze_time

from src._constants import TESTS_DIR
from src.entities import OutputPayload
from tests import redis_client, Postprocess
from tests.tools import get_postproc_input

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestPostProcess(unittest.TestCase):
    keys: list[str] = ["service-mesh", "1", "2"]

    def setUp(self) -> None:
        redis_client.flushall()

    @freeze_time("2022-02-20 12:00:00")
    def test_postprocess(self):
        postproc_input = get_postproc_input(self.keys, STREAM_DATA_PATH)
        msg = Postprocess().run(self.keys, postproc_input)[0]
        payload = OutputPayload(**orjson.loads(msg.value.decode("utf-8")))
        self.assertIsInstance(payload, OutputPayload)
        self.assertTrue(payload.unified_anomaly)
        self.assertGreater(payload.metadata["model_version"], 0)
        for metric, metric_data in payload.data.items():
            self.assertTrue(metric_data)

    def test_preprocess_prev_stale_model(self):
        postproc_input = get_postproc_input(self.keys, STREAM_DATA_PATH, prev_model_stale=True)
        msg = Postprocess().run(self.keys, postproc_input)[0]
        payload = OutputPayload(**orjson.loads(msg.value.decode("utf-8")))
        self.assertIsInstance(payload, OutputPayload)
        self.assertTrue(payload.unified_anomaly)
        self.assertGreater(payload.metadata["model_version"], 0)
        for metric, metric_data in payload.data.items():
            self.assertTrue(metric_data)

    def test_preprocess_no_prev_clf(self):
        postproc_input = get_postproc_input(self.keys, STREAM_DATA_PATH, prev_clf_exists=False)
        msg = Postprocess().run(self.keys, postproc_input)[0]
        payload = OutputPayload(**orjson.loads(msg.value.decode("utf-8")))
        self.assertIsInstance(payload, OutputPayload)
        self.assertTrue(payload.unified_anomaly)
        self.assertEqual(payload.metadata["model_version"], -1)
        for metric, metric_data in payload.data.items():
            self.assertTrue(metric_data)


if __name__ == "__main__":
    unittest.main()
