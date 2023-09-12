import logging
import os
import unittest
from datetime import datetime
from unittest.mock import patch, Mock

from fakeredis import FakeServer, FakeStrictRedis
from omegaconf import OmegaConf
from orjson import orjson
from pynumaflow.function import Datum, DatumMetadata

from numalogic._constants import TESTS_DIR
from numalogic.registry import RedisRegistry
from numalogic.tools.exceptions import ModelKeyNotFound
from numalogic.udfs._config import PipelineConf
from numalogic.udfs.entities import Status, Header, StreamPayload
from numalogic.udfs.preprocess import PreprocessUDF
from tests.udfs.utility import input_json_from_file, store_in_redis

logging.basicConfig(level=logging.DEBUG)
REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM = input_json_from_file(os.path.join(TESTS_DIR, "udfs", "resources", "data", "stream.json"))

DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
    "metadata": DatumMetadata("1", 1),
}


class TestPreprocessUDF(unittest.TestCase):
    def setUp(self) -> None:
        self.preproc_factory = None
        self.registry = RedisRegistry(REDIS_CLIENT)
        _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
        _given_conf_2 = OmegaConf.load(
            os.path.join(TESTS_DIR, "udfs", "resources", "_config2.yaml")
        )
        schema = OmegaConf.structured(PipelineConf)
        pl_conf = PipelineConf(**OmegaConf.merge(schema, _given_conf))
        pl_conf_2 = PipelineConf(**OmegaConf.merge(schema, _given_conf_2))
        store_in_redis(pl_conf, self.registry)
        store_in_redis(pl_conf_2, self.registry)
        self.udf1 = PreprocessUDF(REDIS_CLIENT, pl_conf=pl_conf)
        self.udf2 = PreprocessUDF(REDIS_CLIENT, pl_conf=pl_conf_2)
        self.udf1.register_conf("druid-config", pl_conf.stream_confs["druid-config"])
        self.udf2.register_conf("druid-config", pl_conf_2.stream_confs["druid-config"])

    def tearDown(self) -> None:
        REDIS_CLIENT.flushall()

    def test_preprocess_load_from_registry(self):
        msgs = self.udf2(
            KEYS,
            DATUM,
        )
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(payload.status, Status.ARTIFACT_FOUND)
        self.assertEqual(payload.header, Header.MODEL_INFERENCE)

    def test_preprocess_load_from_config(self):
        msgs = self.udf1(
            KEYS,
            DATUM,
        )

        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(payload.status, Status.ARTIFACT_FOUND)
        self.assertEqual(payload.header, Header.MODEL_INFERENCE)

    @patch.object(RedisRegistry, "load", Mock(side_effect=Exception))
    def test_preprocess_model_not_found(self):
        msgs = self.udf2(KEYS, DATUM)
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    @patch.object(RedisRegistry, "load", Mock(side_effect=ModelKeyNotFound))
    def test_preprocess_4(self):
        msgs = self.udf2(KEYS, DATUM)
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    def test_preprocess_key_error(self):
        with self.assertRaises(KeyError):
            self.udf1(
                KEYS,
                Datum(
                    keys=["service-mesh", "1", "2"],
                    value='{ "uuid": "1"}',
                    **DATUM_KW,
                ),
            )

    @patch.object(PreprocessUDF, "compute", Mock(side_effect=RuntimeError))
    def test_preprocess_run_time_error(self):
        msg = self.udf1(
            KEYS,
            DATUM,
        )
        payload = StreamPayload(**orjson.loads(msg[0].value))
        self.assertEqual(Header.TRAIN_REQUEST, payload.header)
        self.assertEqual(Status.RUNTIME_ERROR, payload.status)


if __name__ == "__main__":
    unittest.main()
