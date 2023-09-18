import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from fakeredis import FakeServer, FakeStrictRedis
from freezegun import freeze_time
from orjson import orjson
from pynumaflow.function import Datum, DatumMetadata

from numalogic.config import NumalogicConf, ModelInfo, TrainerConf, LightningTrainerConf
from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import RedisRegistry, ArtifactData
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.udfs import StreamConf, InferenceUDF
from numalogic.udfs.entities import StreamPayload, Header, Status

REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
    "metadata": DatumMetadata("1", 1),
}
DATA = {
    "uuid": "dd7dfb43-532b-49a3-906e-f78f82ad9c4b",
    "config_id": "conf1",
    "composite_keys": ["service-mesh", "1", "2"],
    "data": [
        [4.801275, 1.4581239],
        [6.0539784, 3.3894842],
        [4.801275, 2.1019106],
        [6.304519, 4.4624624],
        [4.3001933, 3.8186753],
        [4.550734, 2.531102],
        [3.0474892, 2.7456975],
        [1.2937039, 3.1748886],
        [2.0453262, 5.106249],
        [1.7947855, 2.1019106],
        [3.0474892, 2.3165061],
        [0.5420817, 0.5997415],
    ],
    "raw_data": [
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
        [0.0, 0.0],
    ],
    "metrics": ["failed", "degraded"],
    "timestamps": [
        1691623200000,
        1691623260000,
        1691623320000,
        1691623380000,
        1691623440000,
        1691623500000,
        1691623560000,
        1691623620000,
        1691623680000,
        1691623740000,
        1691623800000,
        1691623860000,
    ],
    "header": "model_inference",
    "metadata": {
        "artifact_versions": {"VanillaAE": "0"},
        "tags": {
            "asset_alias": "some-alias",
            "asset_id": "362557362191815079",
            "env": "prd",
        },
    },
}


class TestInferenceUDF(unittest.TestCase):
    def setUp(self) -> None:
        self.udf = InferenceUDF(REDIS_CLIENT)
        self.udf.register_conf(
            "conf1",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )

    @patch.object(
        RedisRegistry,
        "load",
        Mock(
            return_value=ArtifactData(
                artifact=VanillaAE(seq_len=12, n_features=2),
                extras=dict(version="0", timestamp=time.time(), source="registry"),
                metadata={},
            )
        ),
    )
    def test_inference(self):
        with freeze_time(datetime.now() + timedelta(hours=7)):
            msgs = self.udf(
                KEYS,
                Datum(
                    keys=KEYS,
                    value=orjson.dumps(DATA),
                    **DATUM_KW,
                ),
            )
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(Header.MODEL_INFERENCE, payload.header)
        self.assertIsNone(payload.status)
        self.assertTupleEqual((12, 2), payload.get_data().shape)

    @patch.object(
        RedisRegistry,
        "load",
        Mock(
            return_value=ArtifactData(
                artifact=VanillaAE(seq_len=12, n_features=2),
                extras=dict(
                    version="1",
                    timestamp=(datetime.now() - timedelta(hours=25)).timestamp(),
                    source="registry",
                ),
                metadata={},
            )
        ),
    )
    def test_inference_stale(self):
        msgs = self.udf(
            KEYS,
            Datum(
                keys=KEYS,
                value=orjson.dumps(DATA),
                **DATUM_KW,
            ),
        )
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(Header.MODEL_INFERENCE, payload.header)
        self.assertEqual(Status.ARTIFACT_STALE, payload.status)
        self.assertTupleEqual((12, 2), payload.get_data().shape)

    def test_inference_train_request(self):
        data = DATA.copy()
        data["header"] = Header.TRAIN_REQUEST.value
        msgs = self.udf(
            KEYS,
            Datum(
                keys=KEYS,
                value=orjson.dumps(data),
                **DATUM_KW,
            ),
        )
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(Header.TRAIN_REQUEST, payload.header)
        self.assertIsNone(payload.status)

    def test_inference_no_artifact(self):
        msgs = self.udf(
            KEYS,
            Datum(
                keys=KEYS,
                value=orjson.dumps(DATA),
                **DATUM_KW,
            ),
        )
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(Header.TRAIN_REQUEST, payload.header)
        self.assertEqual(Status.ARTIFACT_NOT_FOUND, payload.status)

    @patch.object(
        RedisRegistry,
        "load",
        Mock(
            return_value=ArtifactData(
                artifact=VanillaAE(seq_len=12, n_features=2),
                extras=dict(
                    version="1",
                    timestamp=(datetime.now() - timedelta(hours=25)).timestamp(),
                    source="registry",
                ),
                metadata={},
            )
        ),
    )
    @patch.object(InferenceUDF, "compute", Mock(side_effect=RuntimeError))
    def test_inference_compute_err(self):
        msgs = self.udf(
            KEYS,
            Datum(
                keys=KEYS,
                value=orjson.dumps(DATA),
                **DATUM_KW,
            ),
        )
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(Header.TRAIN_REQUEST, payload.header)
        self.assertEqual(Status.RUNTIME_ERROR, payload.status)

    @patch.object(
        RedisRegistry,
        "load",
        Mock(side_effect=RedisRegistryError()),
    )
    def test_redis_registry_err(self):
        msgs = self.udf(
            KEYS,
            Datum(
                keys=KEYS,
                value=orjson.dumps(DATA),
                **DATUM_KW,
            ),
        )
        self.assertEqual(1, len(msgs))
        payload = StreamPayload(**orjson.loads(msgs[0].value))
        self.assertEqual(Header.TRAIN_REQUEST, payload.header)
        self.assertEqual(Status.ARTIFACT_NOT_FOUND, payload.status)


if __name__ == "__main__":
    unittest.main()
