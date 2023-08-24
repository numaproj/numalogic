import logging
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

import numpy as np
from fakeredis import FakeServer, FakeStrictRedis
from freezegun import freeze_time
from omegaconf import OmegaConf
from orjson import orjson

from numalogic._constants import TESTS_DIR
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.registry import RedisRegistry, LocalLRUCache
from numalogic.tools.exceptions import ModelKeyNotFound
from numalogic.udfs._config import StreamConf
from numalogic.udfs.entities import Header, TrainerPayload
from numalogic.udfs.inference import InferenceUDF
from numalogic.udfs.postprocess import PostProcessUDF
from numalogic.udfs.preprocess import PreprocessUDF
from tests.udfs.utility import input_json_from_file, store_in_redis

logging.basicConfig(level=logging.DEBUG)
REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM = input_json_from_file(os.path.join(TESTS_DIR, "udfs", "resources", "data", "stream.json"))


class TestPostProcessUDF(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = RedisRegistry(REDIS_CLIENT)
        self.cache = LocalLRUCache()
        _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
        _given_conf_2 = OmegaConf.load(
            os.path.join(TESTS_DIR, "udfs", "resources", "_config2.yaml")
        )
        schema = OmegaConf.structured(StreamConf)
        self.stream_conf1 = StreamConf(**OmegaConf.merge(schema, _given_conf))
        self.stream_conf2 = StreamConf(**OmegaConf.merge(schema, _given_conf_2))

    def tearDown(self) -> None:
        REDIS_CLIENT.flushall()
        self.cache.clear()

    def test_postprocess_thresh_model_absent(self):
        self.registry.save(KEYS, ["SparseVanillaAE"], SparseVanillaAE(seq_len=12, n_features=2))
        msg = PreprocessUDF(REDIS_CLIENT, stream_conf=self.stream_conf1)(KEYS, DATUM)
        msg = InferenceUDF(REDIS_CLIENT, stream_confs={"druid-config": self.stream_conf1})(
            KEYS, msg[0]
        )
        msg = PostProcessUDF(REDIS_CLIENT, stream_conf=self.stream_conf1)(KEYS, msg[0])
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    def test_postprocess_infer_model_absent(self):
        msg = PreprocessUDF(REDIS_CLIENT, stream_conf=self.stream_conf1)(KEYS, DATUM)
        msg = InferenceUDF(REDIS_CLIENT, stream_confs={"druid-config": self.stream_conf1})(
            KEYS, msg[0]
        )
        msg = PostProcessUDF(REDIS_CLIENT, stream_conf=self.stream_conf1)(KEYS, msg[0])
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    def test_postprocess_preproc_model_absent(self):
        store_in_redis(self.stream_conf2, self.registry)
        msg = PreprocessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, DATUM)
        msg = InferenceUDF(REDIS_CLIENT, stream_confs={"druid-config": self.stream_conf2})(
            KEYS, msg[0]
        )
        msg = PostProcessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, msg[0])
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    def test_postprocess_infer_model_stale(self):
        store_in_redis(self.stream_conf2, self.registry)
        self.registry.save(KEYS, ["SparseVanillaAE"], SparseVanillaAE(seq_len=12, n_features=2))
        self.registry.save(
            KEYS, ["StdDevThreshold"], StdDevThreshold().fit(np.asarray([[0, 1], [1, 2]]))
        )
        with freeze_time(datetime.now() + timedelta(hours=125)):
            msg = PreprocessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, DATUM)
            msg = InferenceUDF(REDIS_CLIENT, stream_confs={"druid-config": self.stream_conf2})(
                KEYS, msg[0]
            )
            msg = PostProcessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, msg[0])
            self.assertEqual(2, len(msg))

    def test_postprocess_all_model_present(self):
        store_in_redis(self.stream_conf2, self.registry)
        self.registry.save(KEYS, ["SparseVanillaAE"], SparseVanillaAE(seq_len=12, n_features=2))
        self.registry.save(
            KEYS, ["StdDevThreshold"], StdDevThreshold().fit(np.asarray([[0, 1], [1, 2]]))
        )
        msg = PreprocessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, DATUM)
        msg = InferenceUDF(REDIS_CLIENT, stream_confs={"druid-config": self.stream_conf2})(
            KEYS, msg[0]
        )
        msg = PostProcessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, msg[0])
        self.assertEqual(1, len(msg))

    def test_postprocess_infer_runtime_error(self):
        store_in_redis(self.stream_conf2, self.registry)
        self.registry.save(KEYS, ["SparseVanillaAE"], SparseVanillaAE(seq_len=12, n_features=1))
        self.registry.save(
            KEYS, ["StdDevThreshold"], StdDevThreshold().fit(np.asarray([[0, 1], [1, 2]]))
        )
        msg = PreprocessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, DATUM)
        msg = InferenceUDF(REDIS_CLIENT, stream_confs={"druid-config": self.stream_conf2})(
            KEYS, msg[0]
        )
        msg = PostProcessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, msg[0])
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    @patch.object(PostProcessUDF, "compute", Mock(side_effect=RuntimeError))
    def test_preprocess_run_time_error(self):
        self.registry.save(KEYS, ["SparseVanillaAE"], SparseVanillaAE(seq_len=12, n_features=2))
        self.registry.save(
            KEYS, ["StdDevThreshold"], StdDevThreshold().fit(np.asarray([[0, 1], [1, 2]]))
        )
        msg = PreprocessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, DATUM)
        msg = InferenceUDF(REDIS_CLIENT, stream_confs={"druid-config": self.stream_conf2})(
            KEYS, msg[0]
        )
        msg = PostProcessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, msg[0])
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)

    @patch.object(RedisRegistry, "load", Mock(side_effect=ModelKeyNotFound))
    def test_preprocess_4(self):
        store_in_redis(self.stream_conf2, self.registry)
        self.registry.save(KEYS, ["SparseVanillaAE"], SparseVanillaAE(seq_len=12, n_features=2))
        self.registry.save(
            KEYS, ["StdDevThreshold"], StdDevThreshold().fit(np.asarray([[0, 1], [1, 2]]))
        )
        msg = PreprocessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, DATUM)
        msg = InferenceUDF(REDIS_CLIENT, stream_confs={"druid-config": self.stream_conf2})(
            KEYS, msg[0]
        )
        msg = PostProcessUDF(REDIS_CLIENT, stream_conf=self.stream_conf2)(KEYS, msg[0])
        payload = TrainerPayload(**orjson.loads(msg[0].value))
        self.assertEqual(payload.header, Header.TRAIN_REQUEST)


if __name__ == "__main__":
    unittest.main()
