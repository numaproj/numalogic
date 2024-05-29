import logging
import os
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from fakeredis import FakeServer, FakeStrictRedis
from omegaconf import OmegaConf
from orjson import orjson
from pynumaflow.mapper import Datum

from numalogic._constants import TESTS_DIR
from numalogic.models.threshold import StdDevThreshold
from registry import RedisRegistry, ArtifactData
from numalogic.udfs import PipelineConf
from numalogic.udfs.entities import Header, TrainerPayload, Status, OutputPayload
from numalogic.udfs.postprocess import PostprocessUDF

logging.basicConfig(level=logging.DEBUG)
REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
}
DATA = {
    "uuid": "dd7dfb43-532b-49a3-906e-f78f82ad9c4b",
    "config_id": "druid-config",
    "pipeline_id": "pipeline1",
    "composite_keys": ["service-mesh", "1", "2"],
    "data": [
        [2.055191, 2.205468],
        [2.4223375, 1.4583645],
        [2.8268616, 2.4160783],
        [2.1107504, 19.458458],
        [2.446076, 2.2556527],
        [2.7057548, 29.579097],
        [3.034152, 25.521946],
        [1.7857871, 100.8762474],
        [1.4797148, 2.4363635],
        [1.526145, 28.6486845],
        [1.0459993, 100.3363016],
        [1.6239338, 100.4365934],
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
    "metrics": ["col1", "col2"],
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
    "status": Status.ARTIFACT_STALE,
    "header": Header.MODEL_INFERENCE,
    "artifact_versions": {"pipeline1:StdDevThreshold": "0", "pipeline1:VanillaAE": "0"},
    "metadata": {
        "tags": {"asset_alias": "data", "asset_id": "123456789", "env": "prd"},
    },
}


@pytest.fixture
def conf():
    _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "tests", "resources", "_config.yaml"))
    schema = OmegaConf.structured(PipelineConf)
    return PipelineConf(**OmegaConf.merge(schema, _given_conf))


@pytest.fixture
def udf(conf):
    yield PostprocessUDF(REDIS_CLIENT, pl_conf=conf)
    REDIS_CLIENT.flushall()


@pytest.fixture
def artifact():
    model = StdDevThreshold().fit(np.asarray([[0, 1], [1, 2]]))
    return ArtifactData(
        artifact=model,
        extras=dict(version="0", timestamp=time.time(), source="test_registry"),
        metadata={},
    )


@pytest.fixture
def bad_artifact():
    model = StdDevThreshold()
    return ArtifactData(
        artifact=model,
        extras=dict(version="0", timestamp=time.time(), source="test_registry"),
        metadata={},
    )


@pytest.fixture
def data(request):
    if request.param == 1:
        return deepcopy(DATA)
    if request.param == 2:
        data = deepcopy(DATA)
        data["pipeline_id"] = "pipeline2"
        data["artifact_versions"] = (
            {"pipeline2:StdDevThreshold": "0", "pipeline2:VanillaAE": "0"},
        )
        return data
    raise ValueError("Invalid param")


@pytest.mark.parametrize("data", [1, 2], indirect=True, ids=["pipeline1", "pipeline2"])
def test_postprocess(udf, mocker, artifact, data):
    mocker.patch.object(RedisRegistry, "load", return_value=artifact)
    msg = udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(data), **DATUM_KW))

    assert len(msg) == 1
    payload = OutputPayload(**orjson.loads(msg[0].value))
    assert payload.unified_anomaly is not None
    assert msg[0].tags == ["output"]
    print(payload)
    assert payload.unified_anomaly <= 10


def test_postprocess_no_artifact(udf):
    msgs = udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(DATA), **DATUM_KW))
    assert len(msgs) == 2
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert payload.header == Header.TRAIN_REQUEST
    assert msgs[0].tags == ["train"]
    assert msgs[1].tags == ["staticthresh"]


def test_postprocess_runtime_err_01(udf, mocker, artifact):
    mocker.patch.object(RedisRegistry, "load", return_value=artifact)
    mocker.patch.object(PostprocessUDF, "compute", side_effect=RuntimeError)
    msgs = udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(DATA), **DATUM_KW))
    assert len(msgs) == 2
    assert msgs[0].tags == ["train"]
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert payload.header == Header.TRAIN_REQUEST
    assert msgs[1].tags == ["staticthresh"]


def test_postprocess_runtime_err_02(udf, mocker, bad_artifact):
    mocker.patch.object(RedisRegistry, "load", return_value=bad_artifact)
    msgs = udf(KEYS, Datum(keys=KEYS, value=orjson.dumps(DATA), **DATUM_KW))
    assert len(msgs) == 2
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert payload.header == Header.TRAIN_REQUEST
    assert msgs[0].tags == ["train"]
    assert msgs[1].tags == ["staticthresh"]


def test_compute(udf, artifact):
    x_inferred = udf.compute(artifact.artifact, np.asarray(DATA["data"]))
    assert x_inferred.shape == (2,)
