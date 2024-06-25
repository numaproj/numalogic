import logging
import os
from datetime import datetime

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from omegaconf import OmegaConf
from orjson import orjson

from pynumaflow.mapper import Datum

from numalogic._constants import TESTS_DIR
from numalogic.udfs import MetricsLoader, PayloadTransformer
from numalogic.udfs._config import PipelineConf
from tests.udfs.utility import input_json_from_file

logging.basicConfig(level=logging.DEBUG)
REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM = input_json_from_file(os.path.join(TESTS_DIR, "udfs", "resources", "data", "stream.json"))

DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
}
MetricsLoader().load_metrics(
    config_file_path=f"{TESTS_DIR}/udfs/resources/numalogic_udf_metrics.yaml"
)

DATA = {
    "uuid": "dd7dfb43-532b-49a3-906e-f78f82ad9c4b",
    "config_id": "druid-config",
    "composite_keys": ["service-mesh", "1", "2"],
    "data": [],
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
    "metadata": {
        "tags": {
            "asset_alias": "some-alias",
            "asset_id": "362557362191815079",
            "env": "prd",
        },
    },
}


@pytest.fixture()
def udf_args():
    return KEYS, Datum(
        keys=KEYS,
        value=orjson.dumps(DATA),
        **DATUM_KW,
    )


@pytest.fixture
def udf():
    _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
    schema = OmegaConf.structured(PipelineConf)
    pl_conf = PipelineConf(**OmegaConf.merge(schema, _given_conf))
    udf = PayloadTransformer(pl_conf=pl_conf)
    udf.register_conf("druid-config", pl_conf.stream_confs["druid-config"])
    yield udf


def test_payloadtx(udf, udf_args):
    msgs = udf(*udf_args)
    assert len(msgs) == 2
    assert orjson.loads(msgs[0].value)["pipeline_id"] == "pipeline1"
    assert orjson.loads(msgs[1].value)["pipeline_id"] == "pipeline2"
