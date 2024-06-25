import os
import time
from typing import Optional

import pandas as pd

from numalogic.config.factory import ConnectorFactory
from numalogic.connectors import DruidFetcherConf
from numalogic.tools.exceptions import ConfigNotFoundError, DruidFetcherError
from numalogic.tools.types import redis_client_t
from numalogic.udfs._config import PipelineConf
from numalogic.udfs._logger import configure_logger
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs.trainer._base import TrainerUDF
from numalogic.udfs._metrics_utility import _increment_counter, _add_summary

METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))
_struct_log = configure_logger()


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

    def register_druid_fetcher_conf(
        self, config_id: str, pipeline_id: str, conf: DruidFetcherConf
    ) -> None:
        """
        Register DruidFetcherConf with the UDF.

        Args:
            config_id: Config ID
            conf: DruidFetcherConf object
        """
        fetcher_id = f"{config_id}-{pipeline_id}"
        self.pl_conf.druid_conf.id_fetcher[fetcher_id] = conf

    def get_druid_fetcher_conf(self, config_id: str, pipeline_id: str) -> DruidFetcherConf:
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
        fetcher_id = f"{config_id}-{pipeline_id}"
        try:
            return self.pl_conf.druid_conf.id_fetcher[fetcher_id]
        except KeyError as err:
            raise ConfigNotFoundError(f"Config with ID {fetcher_id} not found!") from err

    def fetch_data(self, payload: TrainerPayload) -> Optional[pd.DataFrame]:
        """
        Fetch data from druid.

        Args:
            payload: TrainerPayload object

        Returns
        -------
            Dataframe
        """
        _start_time = time.perf_counter()
        logger = _struct_log.bind(udf_vertex=self._vtx)

        _metric_label_values = {
            "source": ":".join(payload.composite_keys),
            "config_id": payload.config_id,
            "pipeline_id": payload.pipeline_id,
            "composite_key": ":".join(payload.composite_keys),
        }
        _stream_conf = self.get_stream_conf(payload.config_id)
        _conf = _stream_conf.ml_pipelines[payload.pipeline_id]
        _fetcher_conf = self.dataconn_conf.fetcher or (
            self.get_druid_fetcher_conf(
                config_id=payload.config_id, pipeline_id=payload.pipeline_id
            )
            if self.dataconn_conf.id_fetcher
            else None
        )
        if not _fetcher_conf:
            raise ConfigNotFoundError(
                f"Druid fetcher config not found for config_id: {payload.config_id},"
                f" pipeline_id: {payload.pipeline_id}!"
            )

        try:
            _df = self.data_fetcher.fetch(
                datasource=_fetcher_conf.datasource,
                filter_keys=_stream_conf.composite_keys,
                filter_values=payload.composite_keys,
                static_filters=_fetcher_conf.static_filters,
                dimensions=list(_fetcher_conf.dimensions),
                delay=self.dataconn_conf.delay_hrs,
                granularity=_fetcher_conf.granularity,
                aggregations=dict(_fetcher_conf.aggregations),
                group_by=list(_fetcher_conf.group_by),
                pivot=_fetcher_conf.pivot,
                hours=_conf.numalogic_conf.trainer.train_hours,
            )
        except DruidFetcherError:
            _increment_counter(
                counter="FETCH_EXCEPTION_COUNTER",
                labels=_metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            logger.exception("Error while fetching data from druid")
            return None
        _end_time = time.perf_counter() - _start_time
        _add_summary(
            "FETCH_TIME_SUMMARY",
            labels=_metric_label_values,
            data=_end_time,
            is_enabled=METRICS_ENABLED,
        )

        logger.info(
            "Fetched data from druid",
            uuid=payload.uuid,
            config_id=payload.config_id,
            pipeline_id=payload.pipeline_id,
            keys=payload.composite_keys,
            df_shape=_df.shape,
            execution_time_ms=round(_end_time * 1000, 4),
        )

        _add_summary(
            "DATAFRAME_SHAPE_SUMMARY",
            labels=_metric_label_values,
            data=_df.shape[0],
        )

        return _df
