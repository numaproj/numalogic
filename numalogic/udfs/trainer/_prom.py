import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz

from numalogic.config.factory import ConnectorFactory
from numalogic.tools.exceptions import ConfigNotFoundError, PrometheusFetcherError
from numalogic.tools.types import redis_client_t
from numalogic.udfs._config import PipelineConf
from numalogic.udfs._logger import configure_logger
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs._metrics_utility import _increment_counter, _add_summary
from numalogic.udfs.trainer._base import TrainerUDF

METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))
_struct_log = configure_logger()


class PromTrainerUDF(TrainerUDF):
    """
    Trainer UDF using Prometheus/Thanos as data source.

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
        self.dataconn_conf = self.pl_conf.prometheus_conf
        data_fetcher_cls = ConnectorFactory.get_cls("PrometheusFetcher")
        try:
            self.data_fetcher = data_fetcher_cls(
                prometheus_server=self.dataconn_conf.url,
                scrape_interval_secs=self.dataconn_conf.scrape_interval,
            )
        except AttributeError as err:
            raise ConfigNotFoundError("Prometheus config not found!") from err

    def fetch_data(self, payload: TrainerPayload) -> Optional[pd.DataFrame]:
        """
        Fetch data from Prometheus/Thanos.

        Args:
            payload: TrainerPayload object

        Returns
        -------
            Dataframe
        """
        _start_time = time.perf_counter()
        logger = _struct_log.bind(udf_vertex=self._vtx)

        _metric_label_values = {
            "config_id": payload.config_id,
            "pipeline_id": payload.pipeline_id,
            "composite_key": ":".join(payload.composite_keys),
            "source": ":".join(payload.composite_keys),
        }
        _stream_conf = self.get_stream_conf(payload.config_id)
        _conf = _stream_conf.ml_pipelines[payload.pipeline_id]

        end_dt = datetime.now(pytz.utc)
        start_dt = end_dt - timedelta(hours=_conf.numalogic_conf.trainer.train_hours)
        metric_name = _conf.metrics[0] if len(_conf.metrics) == 1 else ""

        try:
            _df = self.data_fetcher.fetch(
                start=start_dt,
                end=end_dt,
                metric_name=metric_name,
                return_labels=["rollouts_pod_template_hash"],
                filters={
                    "numalogic": "true",
                    "numalogic_opex_tags": "",
                    **dict(zip(_stream_conf.composite_keys, payload.composite_keys)),
                },
            )
        except PrometheusFetcherError:
            _increment_counter(
                counter="FETCH_EXCEPTION_COUNTER",
                labels=_metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            logger.exception("Error while fetching data from Prometheus", uuid=payload.uuid)
            return None
        _end_time = time.perf_counter() - _start_time
        _add_summary(
            "FETCH_TIME_SUMMARY",
            labels=_metric_label_values,
            data=_end_time,
            is_enabled=METRICS_ENABLED,
        )
        logger.info(
            "Fetched data from Prometheus",
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
            is_enabled=METRICS_ENABLED,
        )
        return _df
