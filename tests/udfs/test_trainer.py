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
from numalogic.connectors.druid import DruidFetcher
from numalogic.tools.exceptions import ConfigNotFoundError
from numalogic.udfs import StreamConf, PipelineConf
from numalogic.udfs.tools import TrainMsgDeduplicator
from numalogic.udfs.trainer import TrainerUDF

REDIS_CLIENT = FakeStrictRedis(server=FakeServer())

logging.basicConfig(level=logging.DEBUG)


def mock_druid_fetch_data(nrows=5000):
    """Mock druid fetch data."""
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "druid.csv"),
        index_col="timestamp",
        nrows=nrows,
    )


class TrainTrainerUDF(unittest.TestCase):
    def setUp(self):
        REDIS_CLIENT.flushall()
        payload = {
            "uuid": "some-uuid",
            "config_id": "druid-config",
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
        conf = OmegaConf.load(os.path.join(TESTS_DIR, "udfs", "resources", "_config.yaml"))
        schema = OmegaConf.structured(PipelineConf)
        conf = OmegaConf.merge(schema, conf)

        self.udf = TrainerUDF(REDIS_CLIENT, pl_conf=OmegaConf.to_object(conf))

    def tearDown(self) -> None:
        REDIS_CLIENT.flushall()

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_01(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(
                        name="VanillaAE", stateful=True, conf={"seq_len": 12, "n_features": 2}
                    ),
                    preprocess=[ModelInfo(name="LogTransformer", stateful=True, conf={})],
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        self.udf(self.keys, self.datum)

        self.assertEqual(
            2,
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::LATEST",
                b"5984175597303660107::StdDevThreshold::LATEST",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_02(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    preprocess=[ModelInfo(name="StandardScaler", conf={})],
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        self.udf(self.keys, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::LATEST",
                b"5984175597303660107::StdDevThreshold::LATEST",
                b"5984175597303660107::StandardScaler::LATEST",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_03(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    preprocess=[ModelInfo(name="LogTransformer"), ModelInfo(name="StandardScaler")],
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        self.udf(self.keys, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::LATEST",
                b"5984175597303660107::StdDevThreshold::LATEST",
                b"5984175597303660107::LogTransformer:StandardScaler::LATEST",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_do_train(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    preprocess=[ModelInfo(name="LogTransformer"), ModelInfo(name="StandardScaler")],
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        time.time()
        self.udf(self.keys, self.datum)
        with freeze_time(datetime.now() + timedelta(days=2)):
            self.udf(self.keys, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::LATEST",
                b"5984175597303660107::StdDevThreshold::LATEST",
                b"5984175597303660107::LogTransformer:StandardScaler::LATEST",
            ),
        )
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::1",
                b"5984175597303660107::StdDevThreshold::1",
                b"5984175597303660107::LogTransformer:StandardScaler::1",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_do_not_train_1(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    preprocess=[ModelInfo(name="LogTransformer"), ModelInfo(name="StandardScaler")],
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        self.udf(self.keys, self.datum)
        self.udf(self.keys, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::LATEST",
                b"5984175597303660107::StdDevThreshold::LATEST",
                b"5984175597303660107::LogTransformer:StandardScaler::LATEST",
            ),
        )
        self.assertEqual(
            0,
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::1",
                b"5984175597303660107::StdDevThreshold::1",
                b"5984175597303660107::LogTransformer:StandardScaler::1",
            ),
        )

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_do_not_train_2(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    preprocess=[ModelInfo(name="LogTransformer"), ModelInfo(name="StandardScaler")],
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        self.udf(self.keys, self.datum)
        ts = datetime.strptime("2022-05-24 10:00:00", "%Y-%m-%d %H:%M:%S")
        with freeze_time(ts + timedelta(hours=25)):
            TrainMsgDeduplicator(REDIS_CLIENT).ack_read(self.keys, "some-uuid")
            print(datetime.now())
        with freeze_time(ts + timedelta(hours=25) + timedelta(minutes=15)):
            self.udf(self.keys, self.datum)
            print(datetime.now())
        self.assertEqual(
            0,
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::1",
                b"5984175597303660107::StdDevThreshold::1",
                b"5984175597303660107::LogTransformer:StandardScaler::1",
            ),
        )

    def test_trainer_conf_err(self):
        udf = TrainerUDF(REDIS_CLIENT)
        with self.assertRaises(ConfigNotFoundError):
            udf(self.keys, self.datum)

    @patch.object(DruidFetcher, "fetch", Mock(return_value=mock_druid_fetch_data(nrows=10)))
    def test_trainer_data_insufficient(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    preprocess=[ModelInfo(name="StandardScaler", conf={})],
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        self.udf(self.keys, self.datum)
        self.assertFalse(
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::LATEST",
                b"5984175597303660107::StdDevThreshold::LATEST",
                b"5984175597303660107::StandardScaler::LATEST",
            )
        )

    @patch.object(DruidFetcher, "fetch", Mock(side_effect=RuntimeError))
    def test_trainer_datafetcher_err(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    preprocess=[ModelInfo(name="StandardScaler", conf={})],
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        self.udf(self.keys, self.datum)
        self.assertFalse(
            REDIS_CLIENT.exists(
                b"5984175597303660107::VanillaAE::LATEST",
                b"5984175597303660107::StdDevThreshold::LATEST",
                b"5984175597303660107::StandardScaler::LATEST",
            )
        )

    def test_TrainMsgDeduplicator(self):
        train_dedup = TrainMsgDeduplicator(REDIS_CLIENT)
        train_dedup.retrain_freq_ts = 10
        train_dedup.retry_ts = 5
        self.assertEqual(train_dedup.retrain_freq, 10 * 60 * 60)
        self.assertEqual(train_dedup.retry, 5)

    @patch("redis.Redis.hset", Mock(side_effect=RedisError))
    def test_TrainMsgDeduplicator_exception_1(self):
        train_dedup = TrainMsgDeduplicator(REDIS_CLIENT)
        train_dedup.ack_read(self.keys, "some-uuid")
        self.assertLogs("RedisError")
        train_dedup.ack_train(self.keys, "some-uuid")
        self.assertLogs("RedisError")

    @patch("redis.Redis.hgetall", Mock(side_effect=RedisError))
    def test_TrainMsgDeduplicator_exception_2(self):
        train_dedup = TrainMsgDeduplicator(REDIS_CLIENT)
        train_dedup.ack_read(self.keys, "some-uuid")
        self.assertLogs("RedisError")


if __name__ == "__main__":
    unittest.main()
