import logging
import time
from typing import Optional

import pandas as pd

from numalogic.config.factory import ConnectorFactory
from numalogic.connectors import DruidFetcherConf
from numalogic.tools.exceptions import ConfigNotFoundError
from numalogic.tools.types import redis_client_t
from numalogic.udfs._config import PipelineConf
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs.trainer._base import TrainerUDF

_LOGGER = logging.getLogger(__name__)


class DruidTrainerUDF(TrainerUDF):
    """
    Trainer UDF using Druid as data source.

    Args:
        r_client: Redis client
        pl_conf: Pipeline config
    """

    def __init__(
        self,
        r_client: redis_client_t,
        pl_conf: Optional[PipelineConf] = None,
    ):
        super().__init__(r_client=r_client, pl_conf=pl_conf)
        self.dataconn_conf = self.pl_conf.druid_conf
        data_fetcher_cls = ConnectorFactory.get_cls("DruidFetcher")
        try:
            self.data_fetcher = data_fetcher_cls(
                url=self.dataconn_conf.url, endpoint=self.dataconn_conf.endpoint
            )
        except AttributeError as err:
            raise ConfigNotFoundError("Druid config not found!") from err

    def register_druid_fetcher_conf(self, config_id: str, conf: DruidFetcherConf) -> None:
        """
        Register DruidFetcherConf with the UDF.

        Args:
            config_id: Config ID
            conf: DruidFetcherConf object
        """
        self.pl_conf.druid_conf.id_fetcher[config_id] = conf

    def get_druid_fetcher_conf(self, config_id: str) -> DruidFetcherConf:
        """
        Get DruidFetcherConf with the given ID.

        Args:
            config_id: Config ID

        Returns
        -------
            DruidFetcherConf object

        Raises
        ------
            ConfigNotFoundError: If config with the given ID is not found
        """
        try:
            return self.pl_conf.druid_conf.id_fetcher[config_id]
        except KeyError as err:
            raise ConfigNotFoundError(f"Config with ID {config_id} not found!") from err

    def fetch_data(self, payload: TrainerPayload) -> pd.DataFrame:
        """
        Fetch data from druid.

        Args:
            payload: TrainerPayload object

        Returns
        -------
            Dataframe
        """
        _start_time = time.perf_counter()
        _conf = self.get_conf(payload.config_id)
        _fetcher_conf = self.dataconn_conf.fetcher or (
            self.get_druid_fetcher_conf(payload.config_id)
            if self.dataconn_conf.id_fetcher
            else None
        )
        if not _fetcher_conf:
            raise ConfigNotFoundError(
                f"Druid fetcher config not found for config_id: {payload.config_id}!"
            )

        try:
            _df = self.data_fetcher.fetch(
                datasource=_fetcher_conf.datasource,
                filter_keys=_conf.composite_keys,
                filter_values=payload.composite_keys,
                dimensions=list(_fetcher_conf.dimensions),
                delay=self.dataconn_conf.delay_hrs,
                granularity=_fetcher_conf.granularity,
                aggregations=dict(_fetcher_conf.aggregations),
                group_by=list(_fetcher_conf.group_by),
                pivot=_fetcher_conf.pivot,
                hours=_conf.numalogic_conf.trainer.train_hours,
            )
        except Exception:
            _LOGGER.exception("%s - Error while fetching data from druid", payload.uuid)
            return pd.DataFrame()

        _LOGGER.debug(
            "%s - Time taken to fetch data: %.3f sec, df shape: %s",
            payload.uuid,
            time.perf_counter() - _start_time,
            _df.shape,
        )
        return _df
