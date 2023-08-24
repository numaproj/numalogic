import logging
import os
import unittest
from datetime import datetime
from unittest.mock import patch, Mock

import pandas as pd
from fakeredis import FakeStrictRedis, FakeServer
from orjson import orjson
from pynumaflow.function import Datum, DatumMetadata

from numalogic._constants import TESTS_DIR
from numalogic.config import NumalogicConf, ModelInfo
from numalogic.config._config import TrainerConf, LightningTrainerConf
from numalogic.connectors._config import DruidConf, DruidFetcherConf
from numalogic.connectors.druid import DruidFetcher
from numalogic.tools.exceptions import ConfigNotFoundError
from numalogic.udfs._config import StreamConf
from numalogic.udfs.trainer import TrainerUDF


logging.basicConfig(level=logging.DEBUG)


REDIS_CLIENT = FakeStrictRedis(server=FakeServer())
KEYS = ["service-mesh", "1", "2"]


def mock_druid_fetch_data(nrows=5000):
    df = pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "druid.csv"),
        index_col="timestamp",
        nrows=nrows,
    )
    return df


class TrainTrainerUDF(unittest.TestCase):
    def setUp(self):
        payload = {
            "uuid": "some-uuid",
            "config_id": "druid-config",
            "composite_keys": ["5984175597303660107"],
            "metrics": ["failed", "degraded"],
        }
        self.datum = Datum(
            keys=KEYS,
            value=orjson.dumps(payload),
            event_time=datetime.now(),
            watermark=datetime.now(),
            metadata=DatumMetadata("1", 1),
        )
        self.udf = TrainerUDF(
            REDIS_CLIENT,
            druid_conf=DruidConf(
                url="http://localhost:8888",
                endpoint="druid/v2",
                fetcher=DruidFetcherConf(datasource="customer-interaction-metrics"),
            ),
        )

    def tearDown(self) -> None:
        REDIS_CLIENT.flushall()

    @patch.object(DruidFetcher, "fetch_data", Mock(return_value=mock_druid_fetch_data()))
    def test_trainer_01(self):
        self.udf.register_conf(
            "druid-config",
            StreamConf(
                numalogic_conf=NumalogicConf(
                    model=ModelInfo(name="VanillaAE", conf={"seq_len": 12, "n_features": 2}),
                    trainer=TrainerConf(pltrainer_conf=LightningTrainerConf(max_epochs=1)),
                )
            ),
        )
        self.udf(KEYS, self.datum)
        self.assertEqual(
            2,
            REDIS_CLIENT.exists(
                "5984175597303660107::VanillaAE::LATEST",
                "5984175597303660107::StdDevThreshold::LATEST",
            ),
        )

    @patch.object(DruidFetcher, "fetch_data", Mock(return_value=mock_druid_fetch_data()))
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
        self.udf(KEYS, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                "5984175597303660107::VanillaAE::LATEST",
                "5984175597303660107::StdDevThreshold::LATEST",
                "5984175597303660107::StandardScaler::LATEST",
            ),
        )

    @patch.object(DruidFetcher, "fetch_data", Mock(return_value=mock_druid_fetch_data()))
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
        self.udf(KEYS, self.datum)
        self.assertEqual(
            3,
            REDIS_CLIENT.exists(
                "5984175597303660107::VanillaAE::LATEST",
                "5984175597303660107::StdDevThreshold::LATEST",
                "5984175597303660107::LogTransformer:StandardScaler::LATEST",
            ),
        )

    def test_trainer_conf_err(self):
        with self.assertRaises(ConfigNotFoundError):
            self.udf(KEYS, self.datum)

    @patch.object(DruidFetcher, "fetch_data", Mock(return_value=mock_druid_fetch_data(nrows=10)))
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
        self.udf(KEYS, self.datum)
        self.assertFalse(
            REDIS_CLIENT.exists(
                "5984175597303660107::VanillaAE::LATEST",
                "5984175597303660107::StdDevThreshold::LATEST",
                "5984175597303660107::StandardScaler::LATEST",
            )
        )

    @patch.object(DruidFetcher, "fetch_data", Mock(side_effect=RuntimeError))
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
        self.udf(KEYS, self.datum)
        self.assertFalse(
            REDIS_CLIENT.exists(
                "5984175597303660107::VanillaAE::LATEST",
                "5984175597303660107::StdDevThreshold::LATEST",
                "5984175597303660107::StandardScaler::LATEST",
            )
        )


if __name__ == "__main__":
    unittest.main()
