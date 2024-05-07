import logging
import os
import unittest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

import pandas as pd
from fakeredis import FakeStrictRedis, FakeServer
from freezegun import freeze_time
from omegaconf import OmegaConf
from orjson import orjson
from pynumaflow.mapper import Datum
from redis import RedisError

from numalogic._constants import TESTS_DIR
from numalogic.config import NumalogicConf, ModelInfo
from numalogic.config import TrainerConf, LightningTrainerConf
from numalogic.connectors import RedisConf, DruidConf, DruidFetcherConf, PrometheusFetcher
from numalogic.connectors.druid import DruidFetcher
from numalogic.tools.exceptions import (
    ConfigNotFoundError,
    DruidFetcherError,
    PrometheusFetcherError,
)
from numalogic.udfs import StreamConf, PipelineConf, MLPipelineConf
from numalogic.udfs.tools import TrainMsgDeduplicator
from numalogic.udfs.trainer import DruidTrainerUDF, PromTrainerUDF

REDIS_CLIENT = FakeStrictRedis(server=FakeServer())

logging.basicConfig(level=logging.DEBUG)


def mock_druid_fetch_data(nrows=5000):
    """Mock druid fetch data."""
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "druid.csv"),
        index_col="timestamp",
        nrows=nrows,
    )


class TestDruidTrainerUDF(unittest.TestCase):
    def setUp(self):
        REDIS_CLIENT.flushall()
        payload = {
            "uuid": "some-uuid",
            "config_id": "druid-config",
            "pipeline_id": "pipeline1",
            "composite_keys": ["5984175597303660107"],
            "metrics": ["failed", "degraded"],
        }
        self.keys = payload["composite_keys"]
        self.datum = Datum(
            keys=self.keys,
            value=orjson.dumps(payload),
            event_time=datetime.now(),
            watermark=datetime.now(),
        )
        conf_1 = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
        schema = OmegaConf.structured(PipelineConf)
        conf_2 = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config2.yaml"))
        conf_1 = OmegaConf.merge(schema, conf_1)
        conf_2 = OmegaConf.merge(schema, conf_2)
        self.pl_conf_1 = PipelineConf(**OmegaConf.merge(schema, conf_1))
        self.pl_conf_2 = PipelineConf(**OmegaConf.merge(schema, conf_2))

        self.udf1 = DruidTrainerUDF(REDIS_CLIENT, pl_conf=OmegaConf.to_object(conf_1))
        self.udf2 = DruidTrainerUDF(REDIS_CLIENT, pl_conf=OmegaConf.to_object(conf_2))

    def tearDown(self) -> None:
        REDIS_CLIENT.flushall()

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_01(self):
        self.udf1(self.keys, self.datum)

        self.assertEqual(
            2,
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_02(self):
        self.udf1(self.keys, self.datum)
        self.assertEqual(
            2,
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_03(self):
        self.udf1.register_conf(
            "druid-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        metrics=["failed", "degraded"],
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
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
        self.udf1(self.keys, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
                b"5984175597303660107::pipeline1:LogTransformer:StandardScaler::LATEST",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_do_train(self):
        self.udf1.register_conf(
            "druid-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        metrics=["failed", "degraded"],
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
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
        self.udf1(self.keys, self.datum)
        with freeze_time(datetime.now() + timedelta(days=2)):
            self.udf1(self.keys, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
                b"5984175597303660107::pipeline1:LogTransformer:StandardScaler::LATEST",
            ),
        )
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::1",
                b"5984175597303660107::pipeline1:StdDevThreshold::1",
                b"5984175597303660107::pipeline1:LogTransformer:StandardScaler::1",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_do_not_train_1(self):
        self.udf1.register_conf(
            "druid-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        metrics=["failed", "degraded"],
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
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
        self.udf1(self.keys, self.datum)
        self.udf1(self.keys, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
                b"5984175597303660107::pipeline1:LogTransformer:StandardScaler::LATEST",
            ),
        )
        self.assertEqual(
            0,
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::1",
                b"5984175597303660107:pipeline1::StdDevThreshold::1",
                b"5984175597303660107:pipeline1::LogTransformer:StandardScaler::1",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_do_not_train_2(self):
        self.udf1.register_conf(
            "druid-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        metrics=["failed", "degraded"],
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
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
        self.udf1(self.keys, self.datum)
        ts = datetime.strptime("2022-05-24 10:00:00", "%Y-%m-%d %H:%M:%S")
        with freeze_time(ts + timedelta(hours=25)):
            TrainMsgDeduplicator(REDIS_CLIENT).ack_read([*self.keys, "pipeline1"], uuid="some-uuid")
        with freeze_time(ts + timedelta(hours=25) + timedelta(minutes=15)):
            self.udf1(self.keys, self.datum)
        self.assertEqual(
            0,
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::1",
                b"5984175597303660107:pipeline1::StdDevThreshold::1",
                b"5984175597303660107:pipeline1::LogTransformer:StandardScaler::1",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_do_not_train_3(self):
        self.udf1.register_conf(
            "druid-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
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
        TrainMsgDeduplicator(REDIS_CLIENT).ack_read([*self.keys, "pipeline1"], "some-uuid")
        ts = datetime.strptime("2022-05-24 10:00:00", "%Y-%m-%d %H:%M:%S")
        with freeze_time(ts + timedelta(minutes=15)):
            self.udf1(self.keys, self.datum)
            self.assertEqual(
                0,
                REDIS_CLIENT.exists(
                    b"5984175597303660107::pipeline1:VanillaAE::0",
                    b"5984175597303660107:pipeline1::StdDevThreshold::0",
                    b"5984175597303660107:pipeline1::LogTransformer:StandardScaler::0",
                ),
            )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data(50)))
    def test_trainer_do_not_train_4(self):
        self.udf1.register_conf(
            "druid-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
                        numalogic_conf=NumalogicConf(
                            model=ModelInfo(
                                name="VanillaAE", conf={"seq_len": 12, "n_features": 2}
                            ),
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
        self.udf1(self.keys, self.datum)
        self.udf1(self.keys, self.datum)

    def test_trainer_conf_err(self):
        with self.assertRaises(ConfigNotFoundError):
            DruidTrainerUDF(
                REDIS_CLIENT,
                pl_conf=PipelineConf(redis_conf=RedisConf(url="redis://localhost:6379", port=0)),
            )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data(nrows=10)))
    def test_trainer_data_insufficient(self):
        self.udf1.register_conf(
            "druid-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
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
        self.udf1(self.keys, self.datum)
        self.assertFalse(
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
                b"5984175597303660107:pipeline1::StandardScaler::LATEST",
            )
        )

    @patch.object(DruidFetcher, "fetch", Mock(side_effect=DruidFetcherError))
    def test_trainer_datafetcher_err(self):
        self.udf1.register_conf(
            "druid-config",
            StreamConf(
                ml_pipelines={
                    "pipeline1": MLPipelineConf(
                        pipeline_id="pipeline1",
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
        self.udf1(self.keys, self.datum)
        self.assertFalse(
            REDIS_CLIENT.exists(
                b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
                b"5984175597303660107:pipeline1::StandardScaler::LATEST",
            )
        )

    @patch.object(
        DruidFetcher, "fetch", Mock(side_effect=[DruidFetcherError, mock_druid_fetch_data()])
    )
    def test_trainer_datafetcher_err_and_train(self):
        ts = datetime.strptime("2022-05-24 10:00:00", "%Y-%m-%d %H:%M:%S")
        with freeze_time(ts):
            self.udf1.register_conf(
                "druid-config",
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
                                trainer=TrainerConf(
                                    pltrainer_conf=LightningTrainerConf(max_epochs=1)
                                ),
                            ),
                        )
                    }
                ),
            )
            self.udf1(self.keys, self.datum)
        with freeze_time(ts + timedelta(minutes=20)):
            self.udf1(self.keys, self.datum)
            self.assertTrue(
                REDIS_CLIENT.exists(
                    b"5984175597303660107::pipeline1:VanillaAE::LATEST",
                    b"5984175597303660107::pipeline1:StdDevThreshold::LATEST",
                    b"5984175597303660107:pipeline1::StandardScaler::LATEST",
                )
            )

    @patch("redis.Redis.hset", Mock(side_effect=RedisError))
    def test_TrainMsgDeduplicator_exception_1(self):
        train_dedup = TrainMsgDeduplicator(REDIS_CLIENT)
        train_dedup.ack_read([*self.keys, "pipeline1"], "some-uuid")
        self.assertLogs("RedisError")
        train_dedup.ack_train([*self.keys, "pipeline1"], "some-uuid")
        self.assertLogs("RedisError")
        train_dedup.ack_insufficient_data([*self.keys, "pipeline1"], "some-uuid", train_records=180)
        self.assertLogs("RedisError")

    @patch("redis.Redis.hset", Mock(side_effect=mock_druid_fetch_data()))
    def test_TrainMsgDeduplicator_insufficent_data(self):
        with self.assertLogs(level="DEBUG") as log:
            train_dedup = TrainMsgDeduplicator(REDIS_CLIENT)
            train_dedup.ack_insufficient_data(
                [*self.keys, "pipeline1"], "some-uuid", train_records=180
            )
        self.assertLogs("Acknowledging insufficient data for the key", log.output[-1])

    @patch("redis.Redis.hgetall", Mock(side_effect=RedisError))
    def test_TrainMsgDeduplicator_exception_2(self):
        train_dedup = TrainMsgDeduplicator(REDIS_CLIENT)
        train_dedup.ack_read([*self.keys, "pipeline1"], "some-uuid")
        self.assertLogs("RedisError")

    def test_druid_from_config_1(self):
        with self.assertLogs(level="WARN") as log:
            self.udf1(self.keys, self.datum)
            self.assertLogs(
                "WARNING:numalogic.udfs._logger:uuid='some-uuid' event='Caught "
                "exception/error while fetching from source' udf_vertex='trainer' "
                "config_id='druid-config' pipeline_id='pipeline1' metadata={} "
                "keys=['5984175597303660107'] level='warning' "
                "timestamp='2024-04-08T16:17:50.554335Z'",
                log.output[-1],
            )

    def test_druid_from_config_2(self):
        with self.assertLogs(level="WARN") as log:
            self.udf2(self.keys, self.datum)
            self.assertLogs(
                "WARNING:numalogic.udfs._logger:uuid='some-uuid' event='Caught "
                "exception/error while fetching from source' udf_vertex='trainer' "
                "config_id='druid-config' pipeline_id='pipeline1' metadata={} "
                "keys=['5984175597303660107'] level='warning'",
                log.output[-1],
            )

    def test_druid_from_config_missing(self):
        pl_conf = PipelineConf(
            stream_confs={
                "druid-config": StreamConf(
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
            druid_conf=DruidConf(url="some-url", endpoint="druid/v2", delay_hrs=3),
        )
        udf3 = DruidTrainerUDF(REDIS_CLIENT, pl_conf=pl_conf)

        self.assertRaises(ConfigNotFoundError, udf3, self.keys, self.datum)

    def test_druid_get_config_error(self):
        pl_conf = PipelineConf(
            stream_confs={
                "druid-config": StreamConf(
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
            druid_conf=DruidConf(
                url="some-url",
                endpoint="druid/v2",
                delay_hrs=3,
                id_fetcher={
                    "some-id-pipeline1": DruidFetcherConf(
                        datasource="some-datasource", dimensions=["some-dimension"]
                    )
                },
            ),
        )
        udf3 = DruidTrainerUDF(REDIS_CLIENT, pl_conf=pl_conf)
        udf3.register_conf("druid-config", pl_conf.stream_confs["druid-config"])
        udf3.register_druid_fetcher_conf(
            "some-id", "pipeline1", pl_conf.druid_conf.id_fetcher["some-id-pipeline1"]
        )
        with self.assertRaises(ConfigNotFoundError):
            udf3.get_druid_fetcher_conf("different-config", "pipeline1")
        with self.assertRaises(ConfigNotFoundError):
            udf3(self.keys, self.datum)


def _mock_mv_fetch_data():
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "prom_mv.csv"),
        index_col="timestamp",
    )


def _mock_default_fetch_data():
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "prom_default.csv"),
        index_col="timestamp",
    )


class TestPrometheusTrainerUDF(unittest.TestCase):
    def setUp(self):
        REDIS_CLIENT.flushall()
        payload = {
            "uuid": "some-uuid",
            "config_id": "odl-graphql",
            "pipeline_id": "pipeline1",
            "composite_keys": ["odl-odlgraphql-usw2-e2e", "odl-graphql"],
            "metrics": [
                "namespace_app_rollouts_cpu_utilization",
                "namespace_app_rollouts_http_request_error_rate",
                "namespace_app_rollouts_memory_utilization",
            ],
        }
        self.keys = payload["composite_keys"]
        self.datum = Datum(
            keys=self.keys,
            value=orjson.dumps(payload),
            event_time=datetime.now(),
            watermark=datetime.now(),
        )
        conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config3.yaml"))
        self.conf = OmegaConf.merge(OmegaConf.structured(PipelineConf), conf)

    @patch.object(PromTrainerUDF, "fetch_data", Mock(return_value=_mock_mv_fetch_data()))
    def test_trainer_01(self):
        udf = PromTrainerUDF(REDIS_CLIENT, pl_conf=OmegaConf.to_object(self.conf))
        udf(self.keys, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"odl-odlgraphql-usw2-e2e:odl-graphql::pipeline1:StandardScaler::LATEST",
                b"odl-odlgraphql-usw2-e2e:odl-graphql::pipeline1:Conv1dVAE::LATEST",
                b"odl-odlgraphql-usw2-e2e:odl-graphql::pipeline1:MahalanobisThreshold::LATEST",
            ),
        )

    @patch.object(PrometheusFetcher, "fetch", Mock(side_effect=PrometheusFetcherError))
    def test_trainer_error(self):
        udf = PromTrainerUDF(REDIS_CLIENT, pl_conf=OmegaConf.to_object(self.conf))
        datum = Datum(
            keys=self.keys,
            value=orjson.dumps(
                {
                    "uuid": "some-uuid",
                    "config_id": "myconf",
                    "pipeline_id": "pipeline1",
                    "composite_keys": [
                        "odl-odlgraphql-usw2-e2e",
                        "odl-graphql",
                    ],
                    "metrics": [
                        "namespace_app_rollouts_http_request_error_rate",
                    ],
                }
            ),
            event_time=datetime.now(),
            watermark=datetime.now(),
        )
        _out = udf(["odl-odlgraphql-usw2-e2e", "odl-graphql"], datum)
        self.assertFalse(
            REDIS_CLIENT.exists(
                b"odl-odlgraphql-usw2-e2e:odl-graphql::pipeline1:StandardScaler::LATEST",
                b"odl-odlgraphql-usw2-e2e:odl-graphql::pipeline1:Conv1dVAE::LATEST",
                b"odl-odlgraphql-usw2-e2e:odl-graphql::pipeline1:MahalanobisThreshold::LATEST",
            )
        )


if __name__ == "__main__":
    unittest.main()
