import logging
import os
from datetime import datetime
from omegaconf import OmegaConf
from orjson import orjson
import pytest

from numalogic._constants import TESTS_DIR
from numalogic.udfs import PipelineConf, MetricsSingleton
from numalogic.udfs.payloadtx import PayloadTransformer
from tests.udfs.utility import input_json_from_file

MetricsSingleton().load_metrics(
    config_file_path=f"{TESTS_DIR}/udfs/resources/numalogic_udf_metrics.yaml"
)
logging.basicConfig(level=logging.DEBUG)
KEYS = ["service-mesh", "1", "2"]
DATUM = input_json_from_file(os.path.join(TESTS_DIR, "udfs", "resources", "data", "stream.json"))

DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
}


@pytest.fixture
def setup():
    _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
    _given_conf_2 = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config2.yaml"))
    schema = OmegaConf.structured(PipelineConf)
    pl_conf = PipelineConf(**OmegaConf.merge(schema, _given_conf))
    pl_conf_2 = PipelineConf(**OmegaConf.merge(schema, _given_conf_2))
    udf1 = PayloadTransformer(pl_conf=pl_conf)
    udf2 = PayloadTransformer(pl_conf=pl_conf_2)
    udf1.register_conf("druid-config", pl_conf.stream_confs["druid-config"])
    udf2.register_conf("druid-config", pl_conf_2.stream_confs["druid-config"])
    return udf1, udf2


def test_pipeline_1(setup):
    msgs = setup[0](KEYS, DATUM)
    assert 2 == len(msgs)
    for msg in msgs:
        data_payload = orjson.loads(msg.value)
        assert data_payload["pipeline_id"]


def test_pipeline_2(setup):
    msgs = setup[1](KEYS, DATUM)
    assert 1 == len(msgs)
    for msg in msgs:
        data_payload = orjson.loads(msg.value)
        assert data_payload["pipeline_id"]
