import logging
import os
from datetime import datetime

from numpy.testing import assert_array_almost_equal
import pytest
from omegaconf import OmegaConf
from orjson import orjson
from pynumaflow.mapper import Datum

from numalogic._constants import TESTS_DIR
from numalogic.udfs import PipelineConf
from numalogic.udfs.entities import Status, Header, OutputPayload
from numalogic.udfs.staticthresh import StaticThresholdUDF

logging.basicConfig(level=logging.DEBUG)
KEYS = ["service-mesh", "1", "2"]
DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
}
DATA = {
    "uuid": "dd7dfb43-532b-49a3-906e-f78f82ad9c4b",
    "config_id": "druid-config",
    "pipeline_id": "pipeline2",
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
    "status": Status.RUNTIME_ERROR,
    "header": Header.MODEL_INFERENCE,
    "artifact_versions": {"pipeline2:StdDevThreshold": "0", "pipeline2:VanillaAE": "0"},
    "metadata": {
        "tags": {"asset_alias": "data", "asset_id": "123456789", "env": "prd"},
    },
}


@pytest.fixture
def conf() -> PipelineConf:
    _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
    schema = OmegaConf.structured(PipelineConf)
    return PipelineConf(**OmegaConf.merge(schema, _given_conf))


@pytest.fixture
def udf(conf) -> StaticThresholdUDF:
    return StaticThresholdUDF(pl_conf=conf)


@pytest.fixture
def udf_args():
    return KEYS, Datum(keys=KEYS, value=orjson.dumps(DATA), **DATUM_KW)


def test_staticthresh(udf, udf_args):
    msgs = udf(*udf_args)
    assert len(msgs) == 1
    assert msgs[0].tags == ["output"]
    payload = OutputPayload(**orjson.loads(msgs[0].value))

    assert_array_almost_equal(payload.unified_anomaly, 1.666, decimal=3)
    assert payload.data
    assert_array_almost_equal(payload.data["col1"], 1.666, decimal=3)
    assert_array_almost_equal(payload.data["col2"], 1.25, decimal=3)
    assert payload.unified_anomaly == payload.data.get("unified_ST")


def test_err_01(udf, udf_args, mocker):
    mocker.patch.object(StaticThresholdUDF, "compute_feature_scores", side_effect=RuntimeError)
    msgs = udf(*udf_args)
    assert len(msgs) == 1
    assert not msgs[0].value


def test_err_02(udf, udf_args, mocker):
    mocker.patch.object(StaticThresholdUDF, "compute_unified_score", side_effect=RuntimeError)
    msgs = udf(*udf_args)
    assert len(msgs) == 1
    assert not msgs[0].value
