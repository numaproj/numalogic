import time
from numalogic.connectors.rds._base import RDSDataFetcher
import pymysql
import pandas as pd
import logging

from numalogic.connectors.utils.aws.config import DatabaseTypes, RDSConfig

_LOGGER = logging.getLogger(__name__)


class MysqlFetcher(RDSDataFetcher):
    """
    MYSQLFetcher that inherits from RDSDataFetcher. It is used to fetch data from a MySQL database.

    The class has several methods:

    - __init__(self, db_config: RDSConfig, **kwargs): Initializes the MYSQLFetcher object with
    the given RDSConfig and additional keyword arguments. - get_connection(self): Establishes a
    connection to the MySQL database using the provided configuration. - get_db_cursor(self,
    connection): Returns a cursor object for executing queries on the database. - execute_query(
    self, query) -> pd.DataFrame: Executes the given query on the database and returns the
    result as a pandas DataFrame.

    The MYSQLFetcher class is designed to be used as a base class for fetching data from a MySQL
    database. It provides methods for establishing a connection, executing queries,
    and retrieving the results. The class can be extended and customized as needed for specific
    use cases.
    """

    database_type = DatabaseTypes.MYSQL.value

    def __init__(self, db_config: RDSConfig, **kwargs):
        super().__init__(db_config)
        self.db_config = db_config
        self.kwargs = kwargs

    def get_connection(self):
        """
        Establishes a connection to the MySQL database using the provided configuration.

        Returns
        -------
            pymysql.connections.Connection: The connection object for the MySQL database.

        Raises
        ------
            None

        Notes: - If SSL/TLS is enabled and configured in the RDSConfig object, the connection
        will be established with SSL/TLS. - If SSL/TLS is not enabled or configured,
        the connection will be established without SSL/TLS. - The connection object is returned
        for further use in executing queries on the database.

        """
        connection = None
        if self.db_config.ssl and self.db_config.ssl_enabled:
            connection = pymysql.connect(
                host=self.db_config.endpoint,
                port=self.db_config.port,
                user=self.db_config.database_username,
                password=self.get_password(),
                db=self.db_config.database_name,
                ssl=self.db_config.ssl.__dict__,
                cursorclass=pymysql.cursors.DictCursor,
                charset="utf8mb4",
                connect_timeout=self.db_config.database_connection_timeout,
            )
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

        Arguments:
        - connection (pymysql.connections.Connection): The connection object for the
        MySQL database.

        Returns
        -------
            pymysql.cursors.Cursor: The cursor object for executing queries on the database.

        Raises
        ------
            None

        Notes
        -----
            - The cursor object is used to execute queries on the database.
            - The connection object must be established before calling this method.

        """
        return connection.cursor()

    def execute_query(self, query) -> pd.DataFrame:
        """
        Executes the given query on the database and returns the result as a pandas DataFrame.

        Arguments:
            query (str): The SQL query to be executed.

        Returns
        -------
            pandas.DataFrame: The result of the query as a DataFrame.

        Notes
        -----
            - This method establishes a connection to the MySQL database using the
            provided configuration.
            - It retrieves a cursor object for executing queries on the database.
            - The query is executed using the cursor object.
            - The result is fetched and converted into a DataFrame.
            - The execution time of the query is logged using the _LOGGER object.
        """
        _start_time = time.perf_counter()
        connection = self.get_connection()
        cursor = self.get_db_cursor(connection)
        cursor.execute(query)
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=col_names)
        _end_time = time.perf_counter() - _start_time
        _LOGGER.info("RDS MYSQL Query: %s, execution time:  %.4fs", query, _end_time)
        return df
