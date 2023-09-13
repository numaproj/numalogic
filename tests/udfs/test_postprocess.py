import logging
import os
import unittest
from copy import deepcopy
from datetime import datetime
from unittest.mock import patch, Mock

import numpy as np
from fakeredis import FakeServer, FakeStrictRedis
from omegaconf import OmegaConf
from orjson import orjson
from pynumaflow.function import DatumMetadata, Datum

from numalogic._constants import TESTS_DIR
from numalogic.models.threshold import StdDevThreshold
from numalogic.registry import RedisRegistry, LocalLRUCache
from numalogic.tools.exceptions import ModelKeyNotFound
from numalogic.udfs._config import PipelineConf
from numalogic.udfs.entities import Header, TrainerPayload, Status
from numalogic.udfs.postprocess import PostprocessUDF

logging.basicConfig(level=logging.DEBUG)
REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
    "metadata": DatumMetadata("1", 1),
}
DATA = {
    "uuid": "dd7dfb43-532b-49a3-906e-f78f82ad9c4b",
    "config_id": "druid-config",
    "composite_keys": ["service-mesh", "1", "2"],
    "data": [
        [2.055191, 2.205468],
        [2.4223375, 1.4583645],
        [2.8268616, 2.4160783],
        [2.1107504, 1.458458],
        [2.446076, 2.2556527],
        [2.7057548, 2.579097],
        [3.034152, 2.521946],
        [1.7857871, 1.8762474],
        [1.4797148, 2.4363635],
        [1.526145, 2.6486845],
        [1.0459993, 1.3363016],
        [1.6239338, 1.4365934],
    ],
    "raw_data": [
        [11.0, 14.0],
        [17.0, 4.0],
        [22.0, 13.0],
        [17.0, 7.0],
        [23.0, 18.0],
        [15.0, 15.0],
        [16.0, 9.0],
        [10.0, 10.0],
        [3.0, 12.0],
        [6.0, 21.0],
        [5.0, 7.0],
        [10.0, 8.0],
    ],
    "metrics": ["failed", "degraded"],
    "timestamps": [
        1691623140000.0,
        1691623200000.0,
        1691623260000.0,
        1691623320000.0,
        1691623380000.0,
        1691623440000.0,
        1691623500000.0,
        1691623560000.0,
        1691623620000.0,
        1691623680000.0,
        1691623740000.0,
        1691623800000.0,
    ],
    "status": "artifact_found",
    "header": "model_inference",
    "metadata": {
        "artifact_versions": {"StdDevThreshold": "0"},
        "tags": {"asset_alias": "data", "asset_id": "123456789", "env": "prd"},
    },
}


class TestPostProcessUDF(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = RedisRegistry(REDIS_CLIENT)
        self.cache = LocalLRUCache()
        _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
        schema = OmegaConf.structured(PipelineConf)
        pl_conf = PipelineConf(**OmegaConf.merge(schema, _given_conf))
        self.udf = PostprocessUDF(REDIS_CLIENT, pl_conf=pl_conf)

    def tearDown(self) -> None:
        REDIS_CLIENT.flushall()
        self.cache.clear()

    def test_postprocess_preproc_artifact_not_found(self):
        msg = self.udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(DATA), **DATUM_KW))

        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    def test_postprocess_inference_model_absent(self):
        data = deepcopy(DATA)
        data["status"] = Status.ARTIFACT_NOT_FOUND
        msg = self.udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(data), **DATUM_KW))
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    def test_postprocess_infer_model_stale(self):
        data = deepcopy(DATA)
        data["status"] = Status.ARTIFACT_STALE
        data["header"] = Header.MODEL_INFERENCE
        self.registry.save(
            KEYS, ["StdDevThreshold"], StdDevThreshold().fit(np.asarray([[0, 1], [1, 2]]))
        )
        msg = self.udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(data), **DATUM_KW))
        self.assertEqual(2, len(msg))

    def test_postprocess_all_model_present(self):
        data = deepcopy(DATA)
        data["status"] = Status.ARTIFACT_FOUND
        data["header"] = Header.MODEL_INFERENCE
        self.registry.save(
            KEYS, ["StdDevThreshold"], StdDevThreshold().fit(np.asarray([[0, 1], [1, 2]]))
        )

        msg = self.udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(data), **DATUM_KW))
        self.assertEqual(1, len(msg))

    @patch("numalogic.udfs.postprocess.PostprocessUDF.compute", Mock(side_effect=RuntimeError))
    def test_postprocess_infer_runtime_error(self):
        msg = self.udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(DATA), **DATUM_KW))
        self.assertEqual(1, len(msg))
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    @patch.object(PostprocessUDF, "compute", Mock(side_effect=RuntimeError))
    def test_preprocess_run_time_error(self):
        msg = self.udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(DATA), **DATUM_KW))
        self.assertEqual(1, len(msg))
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    @patch.object(RedisRegistry, "load", Mock(side_effect=ModelKeyNotFound))
    def test_preprocess_4(self):
        msg = self.udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(DATA), **DATUM_KW))
        self.assertEqual(1, len(msg))
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)


if __name__ == "__main__":
    unittest.main()
