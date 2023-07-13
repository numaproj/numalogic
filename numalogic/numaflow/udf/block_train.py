import logging
import os

import pandas as pd
from omegaconf import OmegaConf
from orjson import orjson
from pynumaflow.function import Datum, Messages, Message
from sklearn.preprocessing import StandardScaler

from numalogic.blocks import (
    PreprocessBlock,
    NNBlock,
    ThresholdBlock,
    PostprocessBlock,
    BlockPipeline,
)
from numalogic.config import (
    NumalogicConf,
    RedisConf,
    DataStreamConf,
    LightningTrainerConf,
    DruidConf,
)
from numalogic.connectors import DruidFetcher
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.numaflow import NumalogicUDF
from numalogic.numaflow.entities import TrainerPayload
from numalogic.registry import RedisRegistry
from numalogic.registry.redis_registry import get_redis_client_from_conf
from numalogic.transforms import TanhNorm

_LOGGER = logging.getLogger(__name__)
REQUEST_EXPIRY = int(os.getenv("REQUEST_EXPIRY") or 300)
MIN_RECORDS = 1000


class TrainBlockUDF(NumalogicUDF):
    """UDF to train the ML model."""

    def __init__(
        self,
        numalogic_conf: NumalogicConf,
        redis_conf: RedisConf,
        stream_conf: DataStreamConf,
        trainer_conf: LightningTrainerConf,
        druid_conf: DruidConf,
    ):
        super().__init__()
        self.conf = numalogic_conf
        self.druid_conf = druid_conf
        self.stream_conf = stream_conf
        self.trainer_conf = trainer_conf
        self._rclient = self._get_redis_client(redis_conf)
        self.model_registry = RedisRegistry(client=self._rclient)
        self._blocks_args = (
            PreprocessBlock(StandardScaler()),
            NNBlock(
                SparseVanillaAE(
                    seq_len=self.stream_conf.window_size, n_features=len(self.stream_conf.metrics)
                ),
                self.stream_conf.window_size,
            ),
            ThresholdBlock(StdDevThreshold()),
            PostprocessBlock(TanhNorm()),
        )
        self._dkeys = ("stdscaler", "sparsevanillae", "stddevthreshold")

    @staticmethod
    def _get_redis_client(redis_conf: RedisConf):
        return get_redis_client_from_conf(redis_conf)

    @staticmethod
    def _drop_message():
        return Messages(Message.to_drop())

    def fetch_data(self, payload: TrainerPayload) -> pd.DataFrame:
        """Fetch the data from the data source."""
        fetcher_conf = self.stream_conf.druid_fetcher
        data_fetcher = DruidFetcher(url=self.druid_conf.url, endpoint=self.druid_conf.endpoint)

        return data_fetcher.fetch_data(
            datasource=fetcher_conf.datasource,
            filter_keys=self.stream_conf.composite_keys + fetcher_conf.dimensions,
            filter_values=payload.composite_keys[1:] + payload.metrics,
            dimensions=OmegaConf.to_container(fetcher_conf.dimensions),
            granularity=fetcher_conf.granularity,
            aggregations=OmegaConf.to_container(fetcher_conf.aggregations),
            group_by=OmegaConf.to_container(fetcher_conf.group_by),
            pivot=fetcher_conf.pivot,
            hours=fetcher_conf.hours,
        )

    def _is_new_request(self, payload: TrainerPayload) -> bool:
        _ckeys = ":".join(*payload.composite_keys, *payload.metrics)
        r_key = f"train::{_ckeys}"
        value = self._rclient.get(r_key)
        if value:
            return False

        self._rclient.setex(r_key, time=REQUEST_EXPIRY, value=1)
        return True

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """Execute the UDF."""
        payload = TrainerPayload(**orjson.loads(datum.value))
        is_new = self._is_new_request(payload)

        if not is_new:
            return self._drop_message()

        try:
            train_df = self.fetch_data(payload)
        except Exception as err:
            _LOGGER.exception(
                "%s - Error while fetching data for keys: %s, metrics: %s, err: %r",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
                err,
            )
            return self._drop_message()

        if len(train_df) < MIN_RECORDS:
            _LOGGER.warning(
                "%s - Insufficient data for training. Expected: %d, Actual: %d",
                payload.uuid,
                MIN_RECORDS,
                len(train_df),
            )
            return self._drop_message()

        block_pl = BlockPipeline(*self._blocks_args, registry=self.model_registry)
        block_pl.fit(train_df.to_numpy())
        block_pl.save(skeys=payload.composite_keys, dkeys=self._dkeys)
        return self._drop_message()
