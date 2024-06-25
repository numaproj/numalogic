import logging
import os
import time
from typing import Optional
import pandas as pd
import hashlib
from numalogic.config.factory import ConnectorFactory
from numalogic.connectors import RDSFetcherConf
from numalogic.tools.exceptions import ConfigNotFoundError, RDSFetcherError
from numalogic.tools.types import redis_client_t
from numalogic.udfs._config import PipelineConf
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs._metrics_utility import _increment_counter, _add_summary
from numalogic.udfs.trainer._base import TrainerUDF
from datetime import datetime, timedelta
import pytz

METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))
_LOGGER = logging.getLogger(__name__)


def get_hash_based_query(config_id: str, filter_keys=list[str], filter_values=list[str]):
    """
    Calculate the hash value based on the given configuration ID and filter keys/values.

    Args:
        - config_id (str): The configuration ID.
        - filter_keys (list[str]): The list of filter keys.
        - filter_values (list[str]): The list of filter values.

    Returns
    -------
    - hash (str): The calculated hash value.

    Raises
    ------
    - RDSFetcherError: If the length of filter_keys and filter_values is not equal.

    """
    filter_pairs = dict(zip(filter_keys, filter_values))
    filter_pairs["config_id"] = config_id
    to_be_hashed_list = [filter_pairs.get(key, "").strip() for key in sorted(filter_pairs.keys())]

    str_to_be_hashed = "".join(to_be_hashed_list)
    result = hashlib.md5(("".join(str_to_be_hashed)).encode(), usedforsecurity=False)
    hash_ = result.hexdigest()
    _LOGGER.info(
        "get_hash_based_query: str_to_be_hashed: %s , "
        "to_be_hashed_list: %s, filter_pairs: %s, hash:%s",
        str_to_be_hashed,
        to_be_hashed_list,
        filter_pairs,
        hash_,
    )
    return hash_


def build_query(
    datasource: str,
    hash_query_type: bool,
    config_id: str,
    dimensions: list[str],
    metrics: list[str],
    filter_keys: list[str],
    filter_values: list[str],
    datetime_column_name: str,
    hash_column_name: str,
    hours: float,
    delay: float,
    reference_dt: Optional[datetime] = None,
) -> str:
    """
    Builds and returns a query string for fetching data from a data source.

    Args:
        datasource (str): The name of the data source.
        hash_query_type (bool): Flag indicating whether to use hash-based query or not.
        config_id (str): The configuration ID.
        dimensions (list[str]): The list of dimensions.
        metrics (list[str]): The list of metrics.
        filter_keys (list[str]): The list of filter keys.
        filter_values (list[str]): The list of filter values.
        datetime_column_name (str): The name of the datetime column.
        hash_column_name (str): The name of the hash column.
        hours (float): The number of hours to fetch data for.
        delay (float): The delay in hours.
        reference_dt (Optional[datetime], optional): The reference datetime. Defaults to None.

    Returns
    -------
        str: The query string.

    Raises
    ------
        RDSFetcherError: If the hash_query_type is False.

    """
    reference_dt = reference_dt or datetime.now(pytz.utc)
    end_dt = reference_dt - timedelta(hours=delay)
    _LOGGER.debug("Querying with end_dt: %s, that is with delay of %s hrs", end_dt, delay)

    start_dt = end_dt - timedelta(hours=hours)

    intervals = (
        f"{datetime_column_name} >= '{start_dt.isoformat()}' "
        f" and {datetime_column_name} <= '{end_dt.isoformat()}'"
    )

    select_columns = [datetime_column_name, *dimensions, *metrics]
    if hash_query_type:
        hash_ = get_hash_based_query(
            config_id, filter_keys=filter_keys, filter_values=filter_values
        )

        return f"""
        select {', '.join(select_columns)}
        from {datasource}
        where {intervals} and {hash_column_name} = '{hash_}'
        """

    raise NotImplementedError("RDS trainer is setup to support hash based query type only")


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
            self.data_fetcher = data_fetcher_cls(db_config=self.dataconn_conf.connection_conf)
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
                f"Config with ID {fetcher_id} not found in rds_conf!"
            ) from err

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

        # TODO : Make this code generic and add to trainer utils
        _config_id = payload.config_id
        _pipeline_id = payload.pipeline_id
        _metric_label_values = {
            "composite_key": ":".join(payload.composite_keys),
            "config_id": _config_id,
            "pipeline_id": _pipeline_id,
            "source": ":".join(payload.composite_keys),
        }
        _stream_conf = self.get_stream_conf(_config_id)
        _conf = _stream_conf.ml_pipelines[_pipeline_id]
        _fetcher_conf = self.dataconn_conf.fetcher or (
            self.get_rds_fetcher_conf(config_id=_config_id, pipeline_id=_pipeline_id)
            if self.dataconn_conf.id_fetcher
            else None
        )
        if not _fetcher_conf:
            raise ConfigNotFoundError(
                f"RDS fetcher config not found for config_id: {_config_id},"
                f" pipeline_id: {_pipeline_id}!"
            )

        query = build_query(
            datasource=_fetcher_conf.datasource,
            dimensions=_fetcher_conf.dimensions,
            metrics=_fetcher_conf.metrics,
            datetime_column_name=_fetcher_conf.datetime_column_name,
            hash_query_type=_fetcher_conf.hash_query_type,
            hash_column_name=_fetcher_conf.hash_column_name,
            config_id=_config_id,
            filter_keys=_stream_conf.composite_keys,
            filter_values=payload.composite_keys,
            hours=_conf.numalogic_conf.trainer.train_hours,
            delay=self.dataconn_conf.delay_hrs,
            reference_dt=datetime.now(),
        )

        try:
            _df = self.data_fetcher.fetch(
                query=query,
                datetime_column_name=_fetcher_conf.datetime_column_name,
                pivot=_fetcher_conf.pivot,
                group_by=list(_fetcher_conf.group_by),
            )
        except RDSFetcherError:
            _increment_counter(
                counter="FETCH_EXCEPTION_COUNTER",
                labels=_metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            _LOGGER.exception("%s - Error while fetching data from RDS", payload.uuid)
            return None
        _end_time = time.perf_counter() - _start_time
        _add_summary(
            "FETCH_TIME_SUMMARY",
            labels=_metric_label_values,
            data=_end_time,
            is_enabled=METRICS_ENABLED,
        )

        _LOGGER.debug(
            "%s - Time taken to fetch data from RDS : %.3f sec, df shape: %s",
            payload.uuid,
            _end_time,
            _df.shape,
        )
        _add_summary(
            "DATAFRAME_SHAPE_SUMMARY",
            labels=_metric_label_values,
            data=_df.shape[0],
            is_enabled=METRICS_ENABLED,
        )
        return _df
