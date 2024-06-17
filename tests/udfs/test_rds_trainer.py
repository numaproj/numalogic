import logging

import pytest
from redis import RedisError
from numalogic.connectors import RDSFetcher
from numalogic.connectors.utils.aws.config import RDSConnectionConfig
from numalogic.tools.exceptions import RDSFetcherError, ConfigNotFoundError
from numalogic.udfs.tools import TrainMsgDeduplicator
from numalogic.udfs.trainer._rds import RDSTrainerUDF, build_query
import re
from datetime import datetime, timedelta
from numalogic._constants import TESTS_DIR
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs._config import load_pipeline_conf
from unittest.mock import patch, Mock
from numalogic.connectors._config import Pivot, RedisConf, RDSConf, RDSFetcherConf
from typing import Optional
import pandas as pd
import os
from fakeredis import FakeStrictRedis, FakeServer
from pynumaflow.mapper import Datum
from orjson import orjson
from omegaconf import OmegaConf
from numalogic.udfs import StreamConf, PipelineConf, MLPipelineConf, MetricsLoader
from numalogic.config import NumalogicConf, ModelInfo
from numalogic.config import TrainerConf, LightningTrainerConf
import time
from freezegun import freeze_time

REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
MetricsLoader().load_metrics(
    config_file_path=f"{TESTS_DIR}/udfs/resources/numalogic_udf_metrics.yaml"
)


# @pytest.fixture
def mock_rds_fetch_data(
    query,
    datetime_column_name: str,
    pivot: Optional[Pivot] = None,
    group_by: Optional[list[str]] = None,
):
    nrows = 5000
    """Mock rds fetch data."""
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "rds.csv"),
        index_col="timestamp",
        nrows=nrows,
    )


def mock_rds_fetch_data_1(nrows=5000):
    """Mock rds fetch data to match druid scenarios."""
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "druid.csv"),
        index_col="timestamp",
        nrows=nrows,
    )


@pytest.fixture()
def mock_trainer_payload():
    return TrainerPayload(
        uuid="979-98789798-98787",
        config_id="fciPluginAppInteractions",
        pipeline_id="metrics",
        composite_keys=["pluginAssetId", "assetId", "interactionName"],
        metrics=["failed", "degraded"],
    )


@pytest.fixture()
def mock_redis_client():
    return FakeStrictRedis(server=FakeServer())


@pytest.fixture()
def mock_pipeline_conf():
    pipeline_conf = f"{TESTS_DIR}/udfs/resources/rds_trainer_config_fetcher_conf.yaml"
    return load_pipeline_conf(pipeline_conf)


@pytest.fixture()
def mock_pipeline_conf_id_fetcher():
    pipeline_conf = f"{TESTS_DIR}/udfs/resources/rds_trainer_config_fetcher_conf1.yaml"
    return load_pipeline_conf(pipeline_conf)


@pytest.fixture()
def mock_RDS_trainer_UDF(mock_pipeline_conf, mock_redis_client):
    return RDSTrainerUDF(mock_redis_client, mock_pipeline_conf)


@pytest.fixture()
def payload():
    return {
        "uuid": "some-uuid",
        "config_id": "rds-config",
        "pipeline_id": "pipeline1",
        "composite_keys": ["5984175597303660107"],
        "metrics": ["failed", "degraded"],
    }


@pytest.fixture()
def udf1(mock_pipeline_conf, payload):
    REDIS_CLIENT.flushall()
    keys = payload["composite_keys"]
    Datum(
        keys=keys,
        value=orjson.dumps(payload),
        event_time=datetime.now(),
        watermark=datetime.now(),
    )
    conf_1 = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "rds_config.yaml"))
    schema = OmegaConf.structured(PipelineConf)
    conf_1 = OmegaConf.merge(schema, conf_1)
    PipelineConf(**OmegaConf.merge(schema, conf_1))
    return RDSTrainerUDF(REDIS_CLIENT, pl_conf=OmegaConf.to_object(conf_1))


@pytest.fixture()
def datum_mock(mock_pipeline_conf, payload):
    REDIS_CLIENT.flushall()
    keys = payload["composite_keys"]
    return Datum(
        keys=keys,
        value=orjson.dumps(payload),
        event_time=datetime.now(),
        watermark=datetime.now(),
    )


@pytest.fixture()
def udf2(mock_pipeline_conf, payload):
    REDIS_CLIENT.flushall()
    keys = payload["composite_keys"]
    Datum(
        keys=keys,
        value=orjson.dumps(payload),
        event_time=datetime.now(),
        watermark=datetime.now(),
    )
    schema = OmegaConf.structured(PipelineConf)
    conf_2 = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config2.yaml"))
    conf_2 = OmegaConf.merge(schema, conf_2)
    PipelineConf(**OmegaConf.merge(schema, conf_2))
    return RDSTrainerUDF(REDIS_CLIENT, pl_conf=OmegaConf.to_object(conf_2))


def test_build_query(mock_trainer_payload, mock_pipeline_conf):
    # assumption of correct functionality
    datetime_str = "04/23/24 13:55:26"
    test_time = datetime.strptime(datetime_str, "%m/%d/%y %H:%M:%S")
    query = build_query(
        "foo",
        True,
        "123",
        ["a", "b"],
        ["c", "d"],
        ["a"],
        ["b"],
        "time",
        "hash",
        2.0,
        1.0,
        reference_dt=test_time,
    ).replace("\n", " ")
    actual_query = re.sub(r"\s+", " ", query)

    expected_query = (
        "select time, a, b, c, d from foo where time >= '2024-04-23T10:55:26' "
        "and time <= '2024-04-23T12:55:26' "
        "and hash = '1692fcfff3e01e7ba8cffc2baadef5f5'"
    )

    assert actual_query.strip() == re.sub(r"\s+", " ", expected_query.replace("\n", " "))

    # assumption of behavior when hash_query_type is False
    with pytest.raises(NotImplementedError):
        build_query(
            "foo", False, "123", ["a", "b"], ["c", "d"], ["a"], ["b"], "time", "hash", 2.0, 1.0
        )


def test_rds_trainer(mock_trainer_payload, mock_pipeline_conf, mock_RDS_trainer_UDF):
    with patch.object(mock_RDS_trainer_UDF.data_fetcher, "fetch", new=mock_rds_fetch_data):
        actual_df = mock_RDS_trainer_UDF.fetch_data(mock_trainer_payload)
        actual_df_count_dict = actual_df.count().to_dict()
        expected_df_count_dict = {"degraded": 4986, "failed": 5000, "success": 5000}
        assert actual_df_count_dict == expected_df_count_dict


def test_rds_trainer_register_rds_fetcher_conf(
    mock_trainer_payload, mock_pipeline_conf_id_fetcher, mock_redis_client
):
    mock_RDS_trainer_UDF_obj = RDSTrainerUDF(mock_redis_client, mock_pipeline_conf_id_fetcher)
    mock_RDS_trainer_UDF_obj.register_rds_fetcher_conf(
        "fciPluginAppInteractions", "metrics", mock_RDS_trainer_UDF
    )

    assert mock_RDS_trainer_UDF_obj.pl_conf.rds_conf.connection_conf.database_username == "root"


def test_get_rds_fetcher_conf(
    mock_trainer_payload, mock_pipeline_conf_id_fetcher, mock_redis_client
):
    mock_RDS_trainer_UDF_obj = RDSTrainerUDF(mock_redis_client, mock_pipeline_conf_id_fetcher)
    with pytest.raises(ConfigNotFoundError):
        mock_RDS_trainer_UDF_obj.get_rds_fetcher_conf("fciPluginAppInteractions", "metrics1")


def test_trainer_do_train1(udf1, mocker, datum_mock, payload):
    mocker.patch.object(RDSFetcher, "fetch", Mock(return_value=mock_rds_fetch_data_1()))
    udf1.register_conf(
        "rds-config",
        StreamConf(
            ml_pipelines={
                "pipeline1": MLPipelineConf(
                    pipeline_id="pipeline1",
                    metrics=["failed", "degraded"],
                    numalogic_conf=NumalogicConf(
                        model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                        preprocess=[
                            ModelInfo(name="LogTransformer"),
                            ModelInfo(name="StandardScaler"),
                        ],
                        trainer=TrainerConf(
                            pltrainer_conf=LightningTrainerConf(max_epochs=1),
                            transforms=[ModelInfo(name="DataClipper", conf={"lower": [0, 0]})],
                        ),
                    ),
                )
            }
        ),
    )
    time.time()

    keys = payload["composite_keys"]

    udf1(keys, datum_mock)
    with freeze_time(datetime.now() + timedelta(days=2)):
        udf1(keys, datum_mock)
    assert 3 == REDIS_CLIENT.exists(
        b"5984175597303660107::pipeline1:VanillaAE::LATEST",
        b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
        b"5984175597303660107::pipeline1:LogTransformer:StandardScaler::LATEST",
    )
    assert 3 == REDIS_CLIENT.exists(
        b"5984175597303660107::pipeline1:VanillaAE::1",
        b"5984175597303660107::pipeline1:StdDevThreshold::1",
        b"5984175597303660107::pipeline1:LogTransformer:StandardScaler::1",
    )


def test_trainer_do_not_train_1(udf1, mocker, payload, datum_mock):
    mocker.patch.object(RDSFetcher, "fetch", Mock(return_value=mock_rds_fetch_data_1()))
    udf1.register_conf(
        "rds-config",
        StreamConf(
            ml_pipelines={
                "pipeline1": MLPipelineConf(
                    pipeline_id="pipeline1",
                    metrics=["failed", "degraded"],
                    numalogic_conf=NumalogicConf(
                        model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                        preprocess=[
                            ModelInfo(name="LogTransformer"),
                            ModelInfo(name="StandardScaler"),
                        ],
                        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                    ),
                )
            }
        ),
    )
    keys = payload["composite_keys"]

    udf1(keys, datum_mock)
    udf1(keys, datum_mock)
    assert 3 == REDIS_CLIENT.exists(
        b"5984175597303660107::pipeline1:VanillaAE::LATEST",
        b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
        b"5984175597303660107::pipeline1:LogTransformer:StandardScaler::LATEST",
    )
    assert 0 == REDIS_CLIENT.exists(
        b"5984175597303660107::pipeline1:VanillaAE::1",
        b"5984175597303660107:pipeline1::StdDevThreshold::1",
        b"5984175597303660107:pipeline1::LogTransformer:StandardScaler::1",
    )


def test_trainer_do_not_train_3(mocker, udf1, datum_mock, payload):
    mocker.patch.object(RDSFetcher, "fetch", Mock(return_value=mock_rds_fetch_data_1()))
    udf1.register_conf(
        "rds-config",
        StreamConf(
            ml_pipelines={
                "pipeline1": MLPipelineConf(
                    pipeline_id="pipeline1",
                    numalogic_conf=NumalogicConf(
                        model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                        preprocess=[
                            ModelInfo(name="LogTransformer"),
                            ModelInfo(name="StandardScaler"),
                        ],
                        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                    ),
                )
            }
        ),
    )

    keys = payload["composite_keys"]

    TrainMsgDeduplicator(REDIS_CLIENT).ack_read(key=[*keys, "pipeline1"], uuid="some-uuid")
    ts = datetime.strptime("2022-05-24 10:00:00", "%Y-%m-%d %H:%M:%S")
    with freeze_time(ts + timedelta(minutes=15)):
        udf1(keys, datum_mock)
        assert 0 == REDIS_CLIENT.exists(
            b"5984175597303660107::pipeline1:VanillaAE::0",
            b"5984175597303660107:pipeline1::StdDevThreshold::0",
            b"5984175597303660107:pipeline1::LogTransformer:StandardScaler::0",
        )


def test_trainer_do_not_train_4(mocker, udf1, payload, datum_mock):
    mocker.patch.object(RDSFetcher, "fetch", Mock(return_value=mock_rds_fetch_data_1(50)))
    udf1.register_conf(
        "rds-config",
        StreamConf(
            ml_pipelines={
                "pipeline1": MLPipelineConf(
                    pipeline_id="pipeline1",
                    numalogic_conf=NumalogicConf(
                        model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                        preprocess=[
                            ModelInfo(name="LogTransformer"),
                            ModelInfo(name="StandardScaler"),
                        ],
                        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                    ),
                )
            }
        ),
    )

    keys = payload["composite_keys"]

    udf1(keys, datum_mock)
    udf1(keys, datum_mock)


def test_trainer_conf_err(mock_RDS_trainer_UDF):
    with pytest.raises(ConfigNotFoundError):
        RDSTrainerUDF(
            REDIS_CLIENT,
            pl_conf=PipelineConf(redis_conf=RedisConf(url="redis://localhost:6379", port=0)),
        )


def test_trainer_data_insufficient(mocker, udf1, datum_mock, payload):
    mocker.patch.object(RDSFetcher, "fetch", Mock(return_value=mock_rds_fetch_data_1(10)))
    udf1.register_conf(
        "rds-config",
        StreamConf(
            ml_pipelines={
                "pipeline1": MLPipelineConf(
                    pipeline_id="pipeline1",
                    numalogic_conf=NumalogicConf(
                        model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                        preprocess=[ModelInfo(name="StandardScaler", conf={})],
                        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                    ),
                )
            }
        ),
    )
    keys = payload["composite_keys"]

    udf1(keys, datum_mock)
    assert (
        REDIS_CLIENT.exists(
            b"5984175597303660107::pipeline1:VanillaAE::LATEST",
            b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
            b"5984175597303660107:pipeline1::StandardScaler::LATEST",
        )
        == 0
    )


def test_trainer_datafetcher_err(mocker, udf1, payload, datum_mock):
    mocker.patch.object(RDSFetcher, "fetch", Mock(side_effect=RDSFetcherError))
    udf1.register_conf(
        "rds-config",
        StreamConf(
            ml_pipelines={
                "pipeline1": MLPipelineConf(
                    pipeline_id="pipeline1",
                    numalogic_conf=NumalogicConf(
                        model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                        preprocess=[ModelInfo(name="StandardScaler", conf={})],
                        trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                    ),
                )
            }
        ),
    )

    keys = payload["composite_keys"]

    udf1(keys, datum_mock)
    assert (
        REDIS_CLIENT.exists(
            b"5984175597303660107::pipeline1:VanillaAE::LATEST",
            b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
            b"5984175597303660107:pipeline1::StandardScaler::LATEST",
        )
        == 0
    )


def test_trainer_datafetcher_err_and_train(mocker, udf1, payload, datum_mock):
    mocker.patch.object(
        RDSFetcher, "fetch", Mock(side_effect=[RDSFetcherError, mock_rds_fetch_data_1()])
    )
    ts = datetime.strptime("2022-05-24 10:00:00", "%Y-%m-%d %H:%M:%S")
    keys = payload["composite_keys"]
    with freeze_time(ts):
        udf1.register_conf(
            "rds-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        metrics=["failed", "degraded"],
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
                            preprocess=[ModelInfo(name="StandardScaler", conf={})],
                            trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                        ),
                    )
                }
            ),
        )
        udf1(keys, datum_mock)
    with freeze_time(ts + timedelta(minutes=20)):
        udf1(keys, datum_mock)

        assert (
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
                b"5984175597303660107:pipeline1::StandardScaler::LATEST",
            )
            == 2
        )


def test_TrainMsgDeduplicator_exception_1(mocker, caplog, payload):
    caplog.set_level(logging.INFO)
    mocker.patch("redis.Redis.hset", Mock(side_effect=RedisError))
    train_dedup = TrainMsgDeduplicator(REDIS_CLIENT)
    keys = payload["composite_keys"]
    train_dedup.ack_read(key=[*keys, "pipeline1"], uuid="some-uuid")
    assert "RedisError" in caplog.text
    train_dedup.ack_train(key=[*keys, "pipeline1"], uuid="some-uuid")
    assert "RedisError" in caplog.text
    train_dedup.ack_insufficient_data(key=[*keys, "pipeline1"], uuid="some-uuid", train_records=180)
    assert "RedisError" in caplog.text


def test_TrainMsgDeduplicator_exception_2(mocker, caplog, payload):
    caplog.set_level(logging.INFO)
    mocker.patch("redis.Redis.hset", Mock(side_effect=RedisError))
    train_dedup = TrainMsgDeduplicator(REDIS_CLIENT)
    keys = payload["composite_keys"]
    train_dedup.ack_read(key=[*keys, "pipeline1"], uuid="some-uuid")
    assert "RedisError" in caplog.text


def test_rds_from_config_missing(datum_mock, payload):
    pl_conf = PipelineConf(
        stream_confs={
            "rds-config": StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
                            preprocess=[
                                ModelInfo(name="LogTransformer"),
                            ],
                            trainer=TrainerConf(
                                pltrainer_conf=LightningTrainerConf(max_epochs=1),
                            ),
                        ),
                    )
                }
            )
        },
        rds_conf=RDSConf(connection_conf=RDSConnectionConfig(), delay_hrs=3),
    )
    udf3 = RDSTrainerUDF(REDIS_CLIENT, pl_conf=pl_conf)
    keys = payload["composite_keys"]
    with pytest.raises(ConfigNotFoundError):
        udf3(keys, datum_mock)


def test_rds_get_config_error(payload, datum_mock):
    pl_conf = PipelineConf(
        stream_confs={
            "rds-config": StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
                            preprocess=[
                                ModelInfo(name="LogTransformer"),
                            ],
                            trainer=TrainerConf(
                                pltrainer_conf=LightningTrainerConf(max_epochs=1),
                            ),
                        ),
                    )
                }
            )
        },
        rds_conf=RDSConf(
            connection_conf=RDSConnectionConfig(),
            delay_hrs=3,
            id_fetcher={
                "some-id-pipeline1": RDSFetcherConf(
                    datasource="some-datasource",
                    dimensions=["some-dimension"],
                    metrics=["some-metric"],
                )
            },
        ),
    )
    udf3 = RDSTrainerUDF(REDIS_CLIENT, pl_conf=pl_conf)
    udf3.register_conf("rds-config", pl_conf.stream_confs["rds-config"])
    udf3.register_rds_fetcher_conf(
        "some-id", "pipeline1", pl_conf.rds_conf.id_fetcher["some-id-pipeline1"]
    )
    with pytest.raises(ConfigNotFoundError):
        udf3.get_rds_fetcher_conf("different-config", "pipeline1")
