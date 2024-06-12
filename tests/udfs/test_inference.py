import time
from datetime import datetime, timedelta

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from freezegun import freeze_time
from orjson import orjson
from pynumaflow.mapper import Datum

from numalogic._constants import TESTS_DIR
from numalogic.config import (
    NumalogicConf,
    ModelInfo,
    TrainerConf,
    LightningTrainerConf,
    ScoreConf,
    ScoreAdjustConf,
)
from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import RedisRegistry, ArtifactData
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.udfs import StreamConf, InferenceUDF, MLPipelineConf, MetricsLoader
from numalogic.udfs.entities import StreamPayload, Header, Status, TrainerPayload

MetricsLoader().load_metrics(
    config_file_path=f"{TESTS_DIR}/udfs/resources/numalogic_udf_metrics.yaml"
)
REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
}
DATA = {
    "uuid": "dd7dfb43-532b-49a3-906e-f78f82ad9c4b",
    "config_id": "conf1",
    "pipeline_id": "pipeline1",
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
    "artifact_versions": {"pipeline1:VanillaAE": "0"},
    "metadata": {
        "tags": {
            "asset_alias": "some-alias",
            "asset_id": "362557362191815079",
            "env": "prd",
        },
    },
}


@pytest.fixture
def udf():
    udf = InferenceUDF(REDIS_CLIENT)
    udf.register_conf(
        "conf1",
        StreamConf(
            ml_pipelines={
                "pipeline1": MLPipelineConf(
                    pipeline_id="pipeline1",
                    numalogic_conf=NumalogicConf(
                        model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                    ),
                )
            }
        ),
    )
    yield udf
    REDIS_CLIENT.flushall()


@pytest.fixture
def udf_with_adjust():
    udf = InferenceUDF(REDIS_CLIENT)
    udf.register_conf(
        "conf1",
        StreamConf(
            ml_pipelines={
                "pipeline1": MLPipelineConf(
                    pipeline_id="pipeline1",
                    numalogic_conf=NumalogicConf(
                        model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 1}),
                        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                        score=ScoreConf(adjust=ScoreAdjustConf(upper_limits={"failed": 20})),
                    ),
                )
            }
        ),
    )
    yield udf
    REDIS_CLIENT.flushall()


@pytest.fixture()
def udf_args():
    return KEYS, Datum(
        keys=KEYS,
        value=orjson.dumps(DATA),
        **DATUM_KW,
    )


@freeze_time(datetime.now() + timedelta(hours=7))
def test_inference_reg(udf, udf_args, mocker):
    mocker.patch.object(
        RedisRegistry,
        "load",
        return_value=ArtifactData(
            artifact=VanillaAE(seq_len=12, n_features=2),
            extras=dict(version="0", timestamp=time.time(), source="registry"),
            metadata={},
        ),
    )
    msgs = udf(*udf_args)
    assert len(msgs) == 1
    payload = StreamPayload(**orjson.loads(msgs[0].value))
    assert Header.MODEL_INFERENCE == payload.header
    assert payload.status == Status.ARTIFACT_FOUND
    assert (12, 2) == payload.get_data().shape
    assert msgs[0].tags == ["postprocess"]


@freeze_time(datetime.now() + timedelta(hours=7))
def test_inference_cache(udf, udf_args, mocker):
    mocker.patch.object(
        RedisRegistry,
        "load",
        return_value=ArtifactData(
            artifact=VanillaAE(seq_len=12, n_features=2),
            extras=dict(version="0", timestamp=time.time(), source="cache"),
            metadata={},
        ),
    )
    msgs = udf(*udf_args)
    print(MetricsLoader().get_metrics())
    assert len(msgs) == 1
    payload = StreamPayload(**orjson.loads(msgs[0].value))
    assert Header.MODEL_INFERENCE == payload.header
    assert payload.status == Status.ARTIFACT_FOUND
    assert (12, 2) == payload.get_data().shape
    assert msgs[0].tags == ["postprocess"]


def test_inference_stale(udf, udf_args, mocker):
    mocker.patch.object(
        RedisRegistry,
        "load",
        return_value=ArtifactData(
            artifact=VanillaAE(seq_len=12, n_features=2),
            extras=dict(
                version="1",
                timestamp=(datetime.now() - timedelta(hours=25)).timestamp(),
                source="registry",
            ),
            metadata={},
        ),
    )
    msgs = udf(*udf_args)
    assert len(msgs) == 2

    trainer_payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert Header.TRAIN_REQUEST == trainer_payload.header
    assert msgs[0].tags == ["train"]

    stream_payload = StreamPayload(**orjson.loads(msgs[1].value))
    assert Header.MODEL_INFERENCE == stream_payload.header
    assert (12, 2) == stream_payload.get_data().shape
    assert msgs[1].tags == ["postprocess"]


def test_inference_no_artifact(udf, udf_args):
    msgs = udf(*udf_args)
    assert len(msgs) == 1
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert Header.TRAIN_REQUEST == payload.header
    assert msgs[0].tags == ["train"]


def test_registry_error(udf, udf_args, mocker):
    mocker.patch.object(RedisRegistry, "load", side_effect=RedisRegistryError())
    msgs = udf(*udf_args)
    assert len(msgs) == 1
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert Header.TRAIN_REQUEST == payload.header
    assert msgs[0].tags == ["train"]


def test_compute_err_01(udf, udf_args, mocker):
    mocker.patch.object(
        RedisRegistry,
        "load",
        return_value=ArtifactData(
            artifact=VanillaAE(seq_len=12, n_features=2),
            extras=dict(version="0", timestamp=time.time(), source="registry"),
            metadata={},
        ),
    )
    mocker.patch.object(InferenceUDF, "compute", side_effect=RuntimeError)
    msgs = udf(*udf_args)
    assert len(msgs) == 1
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert Header.TRAIN_REQUEST == payload.header
    assert msgs[0].tags == ["train"]


def test_compute_err_02(udf_with_adjust, udf_args, mocker):
    mocker.patch.object(
        RedisRegistry,
        "load",
        return_value=ArtifactData(
            artifact=VanillaAE(seq_len=12, n_features=2),
            extras=dict(version="0", timestamp=time.time(), source="registry"),
            metadata={},
        ),
    )
    mocker.patch.object(InferenceUDF, "compute", side_effect=RuntimeError)
    msgs = udf_with_adjust(*udf_args)
    assert len(msgs) == 2
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert Header.TRAIN_REQUEST == payload.header
    assert msgs[0].tags == ["train"]
    assert msgs[1].tags == ["staticthresh"]


def test_model_pass_error_01(udf, udf_args, mocker):
    mocker.patch.object(
        RedisRegistry,
        "load",
        return_value=ArtifactData(
            artifact=VanillaAE(seq_len=10, n_features=1),
            extras=dict(version="0", timestamp=time.time(), source="registry"),
            metadata={},
        ),
    )
    msgs = udf(*udf_args)
    assert len(msgs) == 1
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert Header.TRAIN_REQUEST == payload.header
    assert msgs[0].tags == ["train"]


def test_model_pass_error_02(udf_with_adjust, udf_args, mocker):
    mocker.patch.object(
        RedisRegistry,
        "load",
        return_value=ArtifactData(
            artifact=VanillaAE(seq_len=10, n_features=1),
            extras=dict(version="0", timestamp=time.time(), source="registry"),
            metadata={},
        ),
    )
    msgs = udf_with_adjust(*udf_args)
    assert len(msgs) == 2
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert Header.TRAIN_REQUEST == payload.header
    assert msgs[0].tags == ["train"]
    assert msgs[1].tags == ["staticthresh"]
