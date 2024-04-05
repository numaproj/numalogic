from numalogic.connectors.rds._base import RDSDataFetcher
import os
import pymysql
import pandas as pd
import logging

from numalogic.connectors.rds._config import DatabaseTypes, RDSConfig
from numalogic.connectors.utils.aws.db_configurations import load_db_conf

_LOGGER = logging.getLogger(__name__)


class MYSQLFetcher(RDSDataFetcher):
    """
    This code snippet defines a class called MYSQLFetcher that inherits from RDSDataFetcher. It is used to fetch data
    from a MySQL database. The class has several methods:

    - __init__(self, db_config: RDSConfig, **kwargs): Initializes the MYSQLFetcher object with the given RDSConfig
    and additional keyword arguments. - get_connection(self): Establishes a connection to the MySQL database using
    the provided configuration. - get_db_cursor(self, connection): Returns a cursor object for executing queries on
    the database. - execute_query(self, query) -> pd.DataFrame: Executes the given query on the database and returns
    the result as a pandas DataFrame.

    The MYSQLFetcher class is designed to be used as a base class for fetching data from a MySQL database. It
    provides methods for establishing a connection, executing queries, and retrieving the results. The class can be
    extended and customized as needed for specific use cases.
    """
    database_type = DatabaseTypes.mysql.value

    def __init__(self, db_config: RDSConfig, **kwargs):
        """
        Initializes the MYSQLFetcher object with the given RDSConfig and additional keyword arguments.

        Parameters:
        - db_config (RDSConfig): The configuration object for the RDS connection.
        - **kwargs: Additional keyword arguments.

        Returns:
        None
        """

        super().__init__(db_config)
        self.db_config = db_config
        self.kwargs = kwargs

    def get_connection(self):
        """
        Establishes a connection to the MySQL database using the provided configuration.

        Returns:
            pymysql.connections.Connection: The connection object for the MySQL database.

        Raises:
            None

        Notes:
            - If SSL/TLS is enabled and configured in the RDSConfig object, the connection will be established with SSL/TLS.
            - If SSL/TLS is not enabled or configured, the connection will be established without SSL/TLS.
            - The connection object is returned for further use in executing queries on the database.
        """
        if self.db_config.ssl and self.db_config.ssl_enabled:
            connection = pymysql.connect(
                host=self.db_config.endpoint,
                port=self.db_config.port,
                user=self.db_config.database_username,
                password=self.get_password(),
                db=self.db_config.database_name,
                ssl=self.db_config.ssl,
                cursorclass=pymysql.cursors.DictCursor,
                charset="utf8mb4",
                connect_timeout=self.db_config.database_connection_timeout,
            )
            return connection
        else:
            connection = pymysql.connect(
                host=self.db_config.endpoint,
                port=self.db_config.port,
                user=self.db_config.database_username,
                password=self.get_password(),
                db=self.db_config.database_name,
                cursorclass=pymysql.cursors.DictCursor,
                charset="utf8mb4",
                connect_timeout=self.db_config.database_connection_timeout,
            )
            return connection

    def get_db_cursor(self, connection):
        """
        Returns a cursor object for executing queries on the database.

        Parameters:
        - connection (pymysql.connections.Connection): The connection object for the MySQL database.

        Returns:
            pymysql.cursors.Cursor: The cursor object for executing queries on the database.

        Raises:
            None

        Notes:
            - The cursor object is used to execute queries on the database.
            - The connection object must be established before calling this method.
        """
        cursor = connection.cursor()
        return cursor

    def execute_query(self, query) -> pd.DataFrame:
        """
        Executes the given query on the database and returns the result as a pandas DataFrame.

        Parameters:
        - query (str): The SQL query to be executed on the database.

        Returns:
            pd.DataFrame: The result of the query as a pandas DataFrame.

        Raises:
            None

        Notes:
            - This method establishes a connection to the database using the get_connection() method.
            - It retrieves a cursor object using the get_db_cursor() method.
            - The query is executed using the cursor.execute() method.
            - The column names are extracted from the cursor.description attribute.
            - The rows are fetched using the cursor.fetchall() method.
            - The result is returned as a pandas DataFrame with the column names as the column headers.
        """
        connection = self.get_connection()
        cursor = self.get_db_cursor(connection)
        cursor.execute(query)
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=col_names)
