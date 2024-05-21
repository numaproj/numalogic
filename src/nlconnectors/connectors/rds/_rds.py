from typing import Optional
from numalogic.connectors._base import DataFetcher
from numalogic.connectors._config import Pivot
from numalogic.connectors.rds._base import format_dataframe
from numalogic.connectors.utils.aws.config import RDSConnectionConfig
import logging
import pandas as pd
from numalogic.connectors.rds.db.factory import RdsFactory
import time

_LOGGER = logging.getLogger(__name__)


class RDSFetcher(DataFetcher):
    """
    class is a subclass of DataFetcher and ABC (Abstract Base Class).
    It is used to fetch data from an RDS (Relational Database Service) instance by executing
    a given SQL query.

    Attributes
    ----------
        db_config (RDSConnectionConfig): The configuration object for the RDS instance.
        fetcher (db.CLASS_TYPE): The fetcher object for the specific database type.

    """

    def __init__(self, db_config: RDSConnectionConfig):
        super().__init__(db_config.endpoint)
        self.db_config = db_config
        factory_object = RdsFactory()
        self.fetcher = factory_object.get_db_handler(db_config.database_type.lower())(db_config)
        _LOGGER.info("Executing for database type: %s", self.fetcher.database_type)

    def fetch(
        self,
        query,
        datetime_column_name: str,
        pivot: Optional[Pivot] = None,
        group_by: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetches data from the RDS instance by executing the given query.

        Args:
            query (str): The SQL query to be executed.
            datetime_column_name (str): The name of the datetime field in the fetched data.
            pivot (Optional[Pivot], optional): The pivot configuration for the fetched data.
            Defaults to None.
            group_by (Optional[list[str]], optional): The list of fields to group the
            fetched data by. Defaults to None.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame containing the fetched data.
        """
        _start_time = time.perf_counter()
        df = self.fetcher.execute_query(query)
        if df.empty or df.shape[0] == 0:
            _LOGGER.warning("No data found for query : %s ", query)
            return pd.DataFrame()

        formatted_df = format_dataframe(
            df,
            query=query,
            datetime_column_name=datetime_column_name,
            pivot=pivot,
            group_by=group_by,
        )
        _end_time = time.perf_counter() - _start_time
        _LOGGER.info("RDS Query: %s Fetch Time: %.4fs", query, _end_time)
        return formatted_df

    def raw_fetch(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError
