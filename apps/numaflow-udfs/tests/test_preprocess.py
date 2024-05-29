import json
import logging
import os
from datetime import datetime

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from omegaconf import OmegaConf
from orjson import orjson

from pynumaflow.mapper import Datum

from numalogic._constants import TESTS_DIR
from registry import RedisRegistry
from numalogic.tools.exceptions import ModelKeyNotFound
from numalogic.udfs._config import PipelineConf
from numalogic.udfs.entities import Status, Header, StreamPayload, TrainerPayload
from numalogic.udfs.preprocess import PreprocessUDF
from tests.test_udfs.utility import input_json_from_file, store_in_redis

logging.basicConfig(level=logging.DEBUG)
REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]
DATUM = input_json_from_file(os.path.join(TESTS_DIR, "tests", "resources", "data", "stream.json"))

DATUM_KW = {
    "event_time": datetime.now(),
    "watermark": datetime.now(),
}


@pytest.fixture
def setup():
    registry = RedisRegistry(REDIS_CLIENT)
    _given_conf = OmegaConf.load(os.path.join(TESTS_DIR, "tests", "resources", "_config.yaml"))
    _given_conf_2 = OmegaConf.load(os.path.join(TESTS_DIR, "tests", "resources", "_config2.yaml"))
    schema = OmegaConf.structured(PipelineConf)
    pl_conf = PipelineConf(**OmegaConf.merge(schema, _given_conf))
    pl_conf_2 = PipelineConf(**OmegaConf.merge(schema, _given_conf_2))
    store_in_redis(pl_conf, registry)
    store_in_redis(pl_conf_2, registry)
    udf1 = PreprocessUDF(REDIS_CLIENT, pl_conf=pl_conf)
    udf2 = PreprocessUDF(REDIS_CLIENT, pl_conf=pl_conf_2)
    udf1.register_conf("druid-config", pl_conf.stream_confs["druid-config"])
    udf2.register_conf("druid-config", pl_conf_2.stream_confs["druid-config"])
    yield udf1, udf2
    REDIS_CLIENT.flushall()


def test_preprocess_load_from_registry(setup):
    _, udf2 = setup
    msgs = udf2(KEYS, DATUM)
    assert len(msgs) == 1
    payload = StreamPayload(**orjson.loads(msgs[0].value))
    assert payload.status == Status.ARTIFACT_FOUND
    assert payload.header == Header.MODEL_INFERENCE


def test_preprocess_load_from_config(setup):
    udf1, _ = setup
    msgs = udf1(KEYS, DATUM)
    assert len(msgs) == 1
    payload = StreamPayload(**orjson.loads(msgs[0].value))
    assert payload.status == Status.ARTIFACT_FOUND
    assert payload.header == Header.MODEL_INFERENCE


def test_preprocess_load_err(setup, mocker):
    mocker.patch.object(RedisRegistry, "load", side_effect=Exception, autospec=True)
    udf1, udf2 = setup
    msgs = udf2(KEYS, DATUM)
    assert len(msgs) == 1
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert payload.header == Header.TRAIN_REQUEST


def test_preprocess_model_not_found(setup, mocker):
    mocker.patch.object(RedisRegistry, "load", side_effect=ModelKeyNotFound, autospec=True)
    _, udf2 = setup
    msgs = udf2(KEYS, DATUM)
    assert len(msgs) == 1
    payload = TrainerPayload(**orjson.loads(msgs[0].value))
    assert payload.header == Header.TRAIN_REQUEST


def test_preprocess_key_error(setup):
    udf1, udf2 = setup
    with pytest.raises(KeyError):
        udf1(KEYS, Datum(keys=["service-mesh", "1", "2"], value='{ "uuid": "1"}', **DATUM_KW))


def test_decode_error(setup):
    udf1, _ = setup
    msgs = udf1(KEYS, Datum(keys=["service-mesh", "1", "2"], value='{ "uuid": "1', **DATUM_KW))
    assert len(msgs) == 1
    assert msgs[0].value == b""


def test_preprocess_run_time_error(setup, mocker):
    mocker.patch.object(PreprocessUDF, "compute", side_effect=RuntimeError)
    udf1, _ = setup
    msg = udf1(KEYS, DATUM)
    assert len(msg) == 2
    payload_1 = TrainerPayload(**orjson.loads(msg[0].value))
    assert payload_1.header == Header.TRAIN_REQUEST
    assert msg[0].tags == ["train"]
    payload_2 = StreamPayload(**orjson.loads(msg[1].value))
    assert msg[1].tags == ["staticthresh"]
    assert payload_2.status == Status.RUNTIME_ERROR


def test_preprocess_data_error(setup):
    udf1, _ = setup
    with open(os.path.join(TESTS_DIR, "tests", "resources", "data", "stream.json"), "rb") as f:
        stream = json.load(f)
    stream["data"] = stream["data"][:5]
    msg = udf1(
        KEYS,
        Datum(
            keys=["service-mesh", "1", "2"],
            value=json.dumps(stream).encode("utf-8"),
            event_time=datetime.now(),
            watermark=datetime.now(),
        ),
    )
    assert len(msg) == 1
    assert msg[0].value == b""
