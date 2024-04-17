import logging
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
from numalogic.udfs._metrics import (
    FETCH_EXCEPTION_COUNTER,
    DATAFRAME_SHAPE_SUMMARY,
    FETCH_TIME_SUMMARY,
    _increment_counter,
    _add_summary,
)
from numalogic.udfs.trainer._base import TrainerUDF
from datetime import datetime, timedelta
import pytz

_LOGGER = logging.getLogger(__name__)


def get_hash_based_query(config_id: str, filter_keys=list[str],
                         filter_values=list[str]):
    if len(filter_keys) != len(filter_values):
        raise RDSFetcherError("filter_keys and filter_values length are not equal")

    filter_pairs = dict(zip(filter_keys, filter_values))
    filter_pairs['config_id'] = config_id
    hash_keys_sorted = sorted(filter_pairs.keys())

    to_be_hashed_list = []
    for key in hash_keys_sorted:
        to_be_hashed_list.append(filter_pairs[key].strip())
    str_to_be_hashed = ''.join(to_be_hashed_list)
    result = hashlib.md5((''.join(str_to_be_hashed)).encode(), usedforsecurity=False)
    hash = result.hexdigest()
    _LOGGER.info(
        "get_hash_based_query: str_to_be_hashed: %s , to_be_hashed_list: %s, filter_pairs: %s, hash:%s",
        str_to_be_hashed, to_be_hashed_list,
        filter_pairs, hash
    )
    return hash


def build_query(datasource: str, hash_query_type: bool, config_id: str, dimensions: list[str],
                metrics: list[str],
                filter_keys: list[str],
                filter_values: list[str],
                datetime_column_name: str,
                hash_column_name: str,
                hours: float,
                delay: float,
                reference_dt: Optional[datetime] = None,
                ) -> str:
    reference_dt = reference_dt or datetime.now(pytz.utc)
    end_dt = reference_dt - timedelta(hours=delay)
    _LOGGER.debug("Querying with end_dt: %s, that is with delay of %s hrs", end_dt, delay)

    start_dt = end_dt - timedelta(hours=hours)

    intervals = f"{datetime_column_name} >= '{start_dt.isoformat()}' and {datetime_column_name} <= '{end_dt.isoformat()}'"

    select_columns = [datetime_column_name] + dimensions + metrics
    if hash_query_type:
        hash = get_hash_based_query(config_id, filter_keys=filter_keys,
                                    filter_values=filter_values)

        # hash = "9bb273c6de2d47bef4eec478f54f9a0c"
        query = f"""
        select {', '.join(select_columns)}
        from {datasource}
        where {intervals} and {hash_column_name} = '{hash}' 
        """
        return query

    raise RDSFetcherError("RDS trainer is setup to support hash based query type only")


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

        try:
            query = build_query(datasource=_fetcher_conf.datasource,
                                dimensions=_fetcher_conf.dimensions,
                                metrics=_fetcher_conf.metrics,
                                datetime_column_name=_fetcher_conf.datetime_column_name,
                                hash_query_type=_fetcher_conf.hash_query_type,
                                hash_column_name=_fetcher_conf.hash_column_name,
                                config_id=_config_id,
                                filter_keys=_stream_conf.composite_keys,
                                filter_values=payload.composite_keys,
                                hours=10240,
                                delay=3.0,
                                reference_dt=datetime.now()
                                )
            _df = self.data_fetcher.fetch(
                query=query,
                datetime_column_name=_fetcher_conf.datetime_column_name,
                pivot=_fetcher_conf.pivot,
                group_by=list(_fetcher_conf.group_by),
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


if __name__ == "__main__":
    ""
    # from numalogic.udfs._config import load_pipeline_conf
    #
    # config = load_pipeline_conf(
    #     "/Users/skondakindi/Desktop/codebase/ml/numalogic/tests/resources/rds_trainer_config_fetcher_conf.yaml")
    # print(config)
    #
    # hash = get_hash_based_query("fciPluginAppInteractions",
    #                             ["pluginAssetId", "assetId", "interactionName"],
    #                             ["12345", "9876", "login"]
    #                             )
    # print(hash)
    #
    # query = build_query(datasource=config.rds_conf.fetcher.datasource,
    #                     hash_query_type=config.rds_conf.fetcher.hash_query_type,
    #                     config_id="fciPluginAppInteractions",
    #                     dimensions=config.rds_conf.fetcher.dimensions,
    #                     metrics=config.rds_conf.fetcher.metrics,
    #                     filter_keys=["pluginAssetId", "assetId", "interactionName"],
    #                     filter_values=["12345", "9876", "login"],
    #                     datetime_column_name=config.rds_conf.fetcher.datetime_column_name,
    #                     hash_column_name=config.rds_conf.fetcher.hash_column_name,
    #                     hours=_conf.numalogic_conf.trainer.train_hours,
    #                     delay=self.dataconn_conf.delay_hrs
    #                     )
    #
    # print(query)
    #
    # from numalogic.connectors.utils.aws.db_configurations import load_db_conf
    # from numalogic.connectors.rds._rds import RDSFetcher
    #
    # # config = load_db_conf("/Users/skondakindi/Desktop/codebase/ml/numalogic/tests/resources/rds_trainer_config_fetcher_conf.yaml")
    # # rds_fetcher = RDSFetcher(config)
    # rds_fetcher = RDSFetcher(config.rds_conf.connection_conf)
    #
    # # query="""select  eventdatetime, cistatus, count from ml_poc.fci_ml_poc13 where hash_assetid_pluginassetid_iname='122d813d0ccfea3ae93a08c6fcebf345' and eventdatetime <='2024-02-28T00:00:00Z' limit 3"""
    # result = rds_fetcher.fetch(query,
    #                            datetime_column_name=config.rds_conf.fetcher.datetime_column_name,
    #                            group_by=config.rds_conf.fetcher.group_by,
    #                            pivot=config.rds_conf.fetcher.pivot)
    # print(result.head())
    # # import time
    # # time.sleep(10)
    #
    # print(query)
    #
    # from numalogic.connectors.druid._druid import DruidFetcher
    # from pydruid.utils.aggregators import doublesum
    # from numalogic.connectors._config import Pivot
    #
    # fetcher = DruidFetcher("https://obelix.odldruid-prd.a.intuit.com/", "druid/v2")
    # df = fetcher.fetch(datasource="tech-ip-customer-interaction-metrics", dimensions=["ciStatus"],
    #                    filter_keys=["assetId"], filter_values=["1084259202722926969"],
    #                    aggregations={"count": doublesum("count")},
    #                    pivot=Pivot(index="timestamp", columns=["ciStatus"], value=["count"], ),
    #                    group_by=["timestamp", "ciStatus"], hours=240, delay=0)
    # print(df.head())
