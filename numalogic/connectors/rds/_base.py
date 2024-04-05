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
        """
        Initialize an instance of the RDSDataFetcher class.

        Parameters:
        - db_config (RDSConfig): The configuration object for the RDS connection.
        - kwargs (dict): Additional keyword arguments.

        Attributes:
        - self.kwargs (dict): Additional keyword arguments.
        - self.db_config (RDSConfig): The configuration object for the RDS connection.
        - self.connection (None): The connection object for the RDS database.
        - self.database_type (str): The type of the database.

        Returns:
        - None
        """
        self.kwargs = kwargs
        self.db_config = db_config
        self.connection = None
        self.database_type = db_config.database_type

    def get_rds_token(self):
        """
        Generates an RDS authentication token using the provided RDS boto3 client.

        Parameters:
            rds_boto3_client (boto3.client): The RDS boto3 client used to generate the authentication token.

        Returns:
            str: The generated RDS authentication token.
        """
        boto3_client_manager = Boto3ClientManager(self.db_config)
        rds_client = boto3_client_manager.get_client(DatabaseServiceProvider.rds.value)
        db_password = boto3_client_manager.get_rds_token(rds_client)
        return db_password

    def get_password(self) -> str:
        """
        Retrieves the password for the RDS connection.

        If 'aws_rds_use_iam' is True, it calls the get_rds_token() method to generate the RDS token.
        Otherwise, it returns the database password from the configuration.

        Returns:
            str: The password for the RDS connection.
        """
        db_password = None
        if self.db_config.aws_rds_use_iam:
            _LOGGER.info("using aws_rds_use_iam to generate RDS Token")
            db_password = self.get_rds_token()
            return db_password
        else:
            _LOGGER.info("using password from config to connect RDS Database")
            db_password = self.db_config.database_password
            return db_password

    def get_connection(self):
        """
        Establishes a connection to the RDS database.

        This method is a placeholder and needs to be implemented in a subclass.
        It should handle the logic for establishing a connection to the RDS database based on the provided configuration.

        Returns:
            None
        """
        pass

    def get_db_cursor(self):
        """
        Retrieves a database cursor for executing queries.

        This method is a placeholder and needs to be implemented in a subclass.
        It should handle the logic for retrieving a database cursor based on the established connection.

        Returns:
            None
        """
        pass

    def execute_query(self, query) -> pd.DataFrame:
        """
        Executes a query on the RDS database and returns the result as a pandas DataFrame.

        Parameters:
            query (str): The SQL query to be executed.

        Returns:
            pd.DataFrame: The result of the query as a pandas DataFrame.
        """
        pass
