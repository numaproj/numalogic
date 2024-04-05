from numalogic.connectors.rds import db
from numalogic.connectors.rds._config import RDSConfig
import logging
import pandas as pd

_LOGGER = logging.getLogger(__name__)


class RDSFetcher(object):

    def __init__(self, db_config: RDSConfig):
        """
        Initialize an instance of the RDSFetcher class.

        Parameters:
            db_config (RDSConfig): The configuration for the RDS instance.

        Returns:
            None

        Raises:
            None
        """
        self.db_config = db_config
        if db.CLASS_TYPE:
            self.fetcher = db.CLASS_TYPE(db_config)
            print(self.fetcher.database_type)

    def fetch(self, query) -> pd.DataFrame:
        """
        Fetches data from the RDS instance by executing the given query.

        Parameters:
            query (str): The SQL query to be executed.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the fetched data.

        """
        return self.fetcher.execute_query(query)
