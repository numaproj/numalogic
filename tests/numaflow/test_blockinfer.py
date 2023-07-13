import logging
import unittest
from datetime import datetime
from unittest.mock import patch, Mock

import numpy as np
from fakeredis import FakeStrictRedis, FakeServer
from orjson import orjson
from pynumaflow.function import Datum, DatumMetadata
from sklearn.preprocessing import StandardScaler

from numalogic.blocks import (
    BlockPipeline,
    PreprocessBlock,
    NNBlock,
    ThresholdBlock,
    PostprocessBlock,
)
from numalogic.config import NumalogicConf, RedisConf, DataStreamConf
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.numaflow.entities import TrainerPayload, OutputPayload
from numalogic.numaflow.udf import InferenceBlockUDF
from numalogic.registry import RedisRegistry
from numalogic.transforms import TanhNorm

PAYLOAD = {
    "data": [
        {"degraded": 0, "failed": 0, "success": 14, "timestamp": 1689189540000},
        {"degraded": 0, "failed": 0, "success": 8, "timestamp": 1689189600000},
        {"degraded": 0, "failed": 0, "success": 6, "timestamp": 1689189660000},
        {"degraded": 0, "failed": 4, "success": 2, "timestamp": 1689189720000},
        {"degraded": 0, "failed": 0, "success": 2, "timestamp": 1689189780000},
        {"degraded": 2, "failed": 0, "success": 4, "timestamp": 1689189840000},
        {"degraded": 0, "failed": 0, "success": 6, "timestamp": 1689190080000},
        {"degraded": 0, "failed": 0, "success": 6, "timestamp": 1689190140000},
        {"degraded": 1, "failed": 0, "success": 2, "timestamp": 1689190500000},
        {"degraded": 0, "failed": 0, "success": 2, "timestamp": 1689190560000},
        {"degraded": 0, "failed": 0, "success": 2, "timestamp": 1689190620000},
        {"degraded": 0, "failed": 0, "success": 2, "timestamp": 1689190680000},
    ],
    "start_time": 1689189540000,
    "end_time": 1689190740000,
    "metadata": {
        "current_pipeline_name": "PluginAsset",
        "druid": {"source": "numalogic"},
        "wavefront": {"source": "numalogic"},
    },
}
MOCK_REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
logging.basicConfig(level=logging.DEBUG)


class TestInferenceBlockUDF(unittest.TestCase):
    def setUp(self) -> None:
        self.skeys = ["PluginAsset", "1804835142986662399"]
        self.dkeys = ["stdscaler", "sparsevanillae", "stddevthreshold"]
        self.datum = Datum(
            keys=self.skeys,
            value=orjson.dumps(PAYLOAD),
            event_time=datetime.now(),
            watermark=datetime.now(),
            metadata=DatumMetadata(msg_id="", num_delivered=0),
        )

    def tearDown(self) -> None:
        MOCK_REDIS_CLIENT.flushall()

    def train_block(self) -> None:
        block_pl = BlockPipeline(
            PreprocessBlock(StandardScaler()),
            NNBlock(SparseVanillaAE(seq_len=12, n_features=3), 12),
            ThresholdBlock(StdDevThreshold()),
            PostprocessBlock(TanhNorm()),
            registry=RedisRegistry(client=MOCK_REDIS_CLIENT),
        )
        rng = np.random.default_rng()
        block_pl.fit(rng.random((1000, 3)), nn__max_epochs=5)
        block_pl.save(skeys=self.skeys, dkeys=self.dkeys)

    @patch.object(InferenceBlockUDF, "_get_redis_client", Mock(return_value=MOCK_REDIS_CLIENT))
    def test_exec_success(self) -> None:
        self.train_block()
        udf = InferenceBlockUDF(
            NumalogicConf(),
            RedisConf(host="localhost", port=6379),
            DataStreamConf(metrics=["degraded", "failed", "success"]),
        )
        out_msgs = udf(self.skeys, self.datum)
        self.assertEqual(1, len(out_msgs))
        self.assertIsInstance(OutputPayload(**orjson.loads(out_msgs[0].value)), OutputPayload)

    @patch.object(InferenceBlockUDF, "_get_redis_client", Mock(return_value=MOCK_REDIS_CLIENT))
    def test_exec_no_model(self) -> None:
        udf = InferenceBlockUDF(
            NumalogicConf(),
            RedisConf(host="localhost", port=6379),
            DataStreamConf(metrics=["degraded", "failed", "success"]),
        )
        out_msgs = udf(self.skeys, self.datum)
        self.assertEqual(1, len(out_msgs))
        self.assertIsInstance(TrainerPayload(**orjson.loads(out_msgs[0].value)), TrainerPayload)


if __name__ == "__main__":
    unittest.main()
