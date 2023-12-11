import logging
import os
import unittest
from datetime import datetime
from fakeredis import FakeServer, FakeStrictRedis
from omegaconf import OmegaConf
from orjson import orjson

from numalogic._constants import TESTS_DIR
from numalogic.udfs._config import PipelineConf
from numalogic.udfs.pipeline import PipelineUDF
from tests.udfs.utility import input_json_from_file

logging.basicConfig(level=logging.DEBUG)
REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM = input_json_from_file(os.path.join(TESTS_DIR, "udfs", "resources", "data", "stream.json"))

DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
}


class TestPipelineUDF(unittest.TestCase):
    def setUp(self) -> None:
        _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
        _given_conf_2 = OmegaConf.load(
            os.path.join(TESTS_DIR, "udfs", "resources", "_config2.yaml")
        )
        schema = OmegaConf.structured(PipelineConf)
        pl_conf = PipelineConf(**OmegaConf.merge(schema, _given_conf))
        pl_conf_2 = PipelineConf(**OmegaConf.merge(schema, _given_conf_2))
        self.udf1 = PipelineUDF(pl_conf=pl_conf)
        self.udf2 = PipelineUDF(pl_conf=pl_conf_2)
        self.udf1.register_conf("druid-config", pl_conf.stream_confs["druid-config"])
        self.udf2.register_conf("druid-config", pl_conf_2.stream_confs["druid-config"])

    def test_pipeline_1(self):
        msgs = self.udf1(KEYS, DATUM)
        self.assertEqual(2, len(msgs))
        for msg in msgs:
            data_payload = orjson.loads(msg.value)
            self.assertTrue(data_payload["pipeline_id"])

    def test_pipeline_2(self):
        msgs = self.udf2(KEYS, DATUM)
        self.assertEqual(1, len(msgs))
        for msg in msgs:
            data_payload = orjson.loads(msg.value)
            self.assertTrue(data_payload["pipeline_id"])


if __name__ == "__main__":
    unittest.main()
