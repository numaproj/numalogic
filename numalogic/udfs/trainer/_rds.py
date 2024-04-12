import logging
import time
from typing import Optional

import pandas as pd

from numalogic.config.factory import ConnectorFactory
from numalogic.connectors import RDSFetcherConf
from numalogic.tools.exceptions import ConfigNotFoundError, RDSFetcherError
from numalogic.tools.types import redis_client_t
from numalogic.udfs._config import PipelineConf
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs._metrics import (
    FETCH_EXCEPTION_COUNTER,
    DATAFRAME_SHAPE_SUMMARY,
    FETCH_TIME_SUMMARY,
    _increment_counter,
    _add_summary,
)
from numalogic.udfs.trainer._base import TrainerUDF

_LOGGER = logging.getLogger(__name__)


def get_hash_based_query(config_id : str):
    ""


class RDSTrainerUDF(TrainerUDF):
    """
    Trainer UDF using RDS as data source.

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
        self.dataconn_conf = self.pl_conf.rds_conf
        data_fetcher_cls = ConnectorFactory.get_cls("RDSFetcher")
        try:
            self.data_fetcher = data_fetcher_cls(
                db_config=self.dataconn_conf.connection_conf
            )
        except AttributeError as err:
            raise ConfigNotFoundError("RDS config not found!") from err

    def register_rds_fetcher_conf(
            self, config_id: str, pipeline_id: str, conf: RDSFetcherConf
    ) -> None:
        """
        Register RDSFetcherConf with the UDF.

        Args:
            config_id: Config ID
            conf: RDSFetcherConf object
        """
        fetcher_id = f"{config_id}-{pipeline_id}"
        self.pl_conf.rds_conf.id_fetcher[fetcher_id] = conf

    def get_rds_fetcher_conf(self, config_id: str, pipeline_id: str) -> RDSFetcherConf:
        """
        Get RDSFetcherConf with the given ID.

        Args:
            config_id: Config ID

        Returns
        -------
            RDSFetcherConf object

        Raises
        ------
            ConfigNotFoundError: If config with the given ID is not found
        """
        fetcher_id = f"{config_id}-{pipeline_id}"
        try:
            return self.pl_conf.rds_conf.id_fetcher[fetcher_id]
        except KeyError as err:
            raise ConfigNotFoundError(
                f"Config with ID {fetcher_id} not found in rds_conf!") from err

    def fetch_data(self, payload: TrainerPayload) -> Optional[pd.DataFrame]:
        """
        Fetch data from RDS.

        Args:
            payload: TrainerPayload object

        Returns
        -------
            Dataframe
        """
        _start_time = time.perf_counter()

        # Make this code generic and add to trainer utils
        _config_id = payload.config_id
        _pipeline_id = payload.pipeline_id
        _metric_label_values = (
            payload.composite_keys,
            ":".join(payload.composite_keys),
            _config_id,
            _pipeline_id,
        )
        _stream_conf = self.get_stream_conf(_config_id)
        _conf = _stream_conf.ml_pipelines[_pipeline_id]
        _fetcher_conf = self.dataconn_conf.fetcher or (
            self.get_rds_fetcher_conf(
                config_id=_config_id, pipeline_id=_pipeline_id
            )
            if self.dataconn_conf.id_fetcher
            else None
        )
        if not _fetcher_conf:
            raise ConfigNotFoundError(
                f"RDS fetcher config not found for config_id: {_config_id},"
                f" pipeline_id: {_pipeline_id}!"
            )

        # datasource=_fetcher_conf.datasource,
        # filter_keys=_stream_conf.composite_keys,
        # filter_values=payload.composite_keys,
        # dimensions=list(_fetcher_conf.dimensions),
        # delay=self.dataconn_conf.delay_hrs,
        # granularity=_fetcher_conf.granularity,
        # aggregations=dict(_fetcher_conf.aggregations),
        # group_by=list(_fetcher_conf.group_by),
        # pivot=_fetcher_conf.pivot,
        # hours=_conf.numalogic_conf.trainer.train_hours,

        try:
            _df = self.data_fetcher.fetch(
                query="",
                datetime_field_name="",
                pivot=_fetcher_conf.pivot,
                group_by=list(_fetcher_conf.group_by,
                              )
            )
        except RDSFetcherError:
            _increment_counter(
                counter=FETCH_EXCEPTION_COUNTER,
                labels=_metric_label_values,
            )
            _LOGGER.exception("%s - Error while fetching data from RDS", payload.uuid)
            return None
        _end_time = time.perf_counter() - _start_time
        _add_summary(
            FETCH_TIME_SUMMARY,
            labels=_metric_label_values,
            data=_end_time,
        )

        _LOGGER.debug(
            "%s - Time taken to fetch data from RDS : %.3f sec, df shape: %s",
            payload.uuid,
            _end_time,
            _df.shape,
        )
        _add_summary(
            DATAFRAME_SHAPE_SUMMARY,
            labels=_metric_label_values,
            data=_df.shape[0],
        )

        return _df
