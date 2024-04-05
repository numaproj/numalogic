import pandas as pd
from numalogic.connectors.rds._config import DatabaseServiceProvider, DatabaseTypes, RDSConfig
from numalogic.connectors.utils.aws.boto3_client_manager import Boto3ClientManager
import logging

_LOGGER = logging.getLogger(__name__)


class RDSDataFetcher(object):
    """
    Class: RDSDataFetcher

    This class represents a data fetcher for RDS (Relational Database Service) connections. It provides methods for retrieving the RDS token, getting the password, establishing a connection, and executing queries.

    Attributes:
    - db_config (RDSConfig): The configuration object for the RDS connection.
    - kwargs (dict): Additional keyword arguments.

    Methods:
    - get_rds_token(): Retrieves the RDS token using the Boto3ClientManager.
    - get_password() -> str: Retrieves the password for the RDS connection. If 'aws_rds_use_iam' is True, it calls the get_rds_token() method, otherwise it returns the database password from the configuration.
    - get_connection(): Placeholder method for establishing a connection to the RDS database.
    - get_db_cursor(): Placeholder method for getting a database cursor.
    - execute_query(query) -> pd.DataFrame: Placeholder method for executing a query and returning the result as a pandas DataFrame.
    """

    def __init__(self, db_config: RDSConfig, **kwargs):
        self.kwargs = kwargs
        self.db_config = db_config
        self.connection = None
        self.database_type = db_config.database_type

    def get_rds_token(self):
        boto3_client_manager = Boto3ClientManager(self.db_config)
        rds_client = boto3_client_manager.get_client(DatabaseServiceProvider.rds.value)
        db_password = boto3_client_manager.get_rds_token(rds_client)
        return db_password

    def get_password(self) -> str:
        db_password = None
        if self.db_config.aws_rds_use_iam:
            _LOGGER.info("using aws_rds_use_iam to generate RDS Token")
            db_password = self.get_rds_token()
            return db_password
        else:
            _LOGGER.info("using password from config to connect RDS Database")
            db_password = self.db_config.database_password
            print(f"db_password: {db_password}")
            return db_password

    def get_connection(self):
        pass

    def get_db_cursor(self):
        pass

    def execute_query(self, query) -> pd.DataFrame:
        pass
