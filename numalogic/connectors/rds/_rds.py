from abc import ABC
from numalogic.connectors._base import DataFetcher
from numalogic.connectors.rds._config import RDSConfig
import logging
import pandas as pd
from numalogic.connectors.rds.db.factory import RdsFactory

_LOGGER = logging.getLogger(__name__)


class RDSFetcher(DataFetcher, ABC):
    """
    RDSFetcher class.

    This class is a subclass of DataFetcher and ABC (Abstract Base Class).
    It is used to fetch data from an RDS (Relational Database Service) instance by executing
    a given SQL query.

    Attributes
    ----------
        db_config (RDSConfig): The configuration object for the RDS instance.
        fetcher (db.CLASS_TYPE): The fetcher object for the specific database type.

    Methods
    -------
        __init__(self, db_config: RDSConfig):
            Initializes the RDSFetcher object with the given RDSConfig object.
        fetch(self, query):
            Fetches data from the RDS instance by executing the given SQL query.

    """

    def __init__(self, db_config: RDSConfig):
        super().__init__(db_config.__dict__.get("url"))
        self.db_config = db_config
        factory_object = RdsFactory()
        self.fetcher = factory_object.get_db_handler(db_config.database_type.lower())
        _LOGGER.info("Executing for database type: %s", self.fetcher.database_type)

    def fetch(self, query) -> pd.DataFrame:
        """
        Fetches data from the RDS instance by executing the given query.

        Parameters
        ----------
            query (str): The SQL query to be executed.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame containing the fetched data.

        """
        return self.fetcher.execute_query(query)

    def raw_fetch(self, *args, **kwargs) -> pd.DataFrame:
        pass
