import time

from pymysql.cursors import Cursor

from numalogic.connectors.rds._base import RDSBase
import pymysql
import pandas as pd
import logging

from numalogic.connectors.utils.aws.config import DatabaseTypes, RDSConnectionConfig

_LOGGER = logging.getLogger(__name__)


class MysqlFetcher(RDSBase):
    """
    class that inherits from RDSBase. It is used to fetch data from a MySQL database.

    - __init__(self, db_config: RDSConnectionConfig, **kwargs): Initializes the MysqlFetcher object
     with the given RDSConnectionConfig and additional keyword arguments.

    The MysqlFetcher class is designed to be used as a base class for fetching data from a MySQL
    database. It provides methods for establishing a connection, executing queries,
    and retrieving the results. The class can be extended and customized as needed for specific
    use cases.
    """

    database_type = DatabaseTypes.MYSQL

    def __init__(self, db_config: RDSConnectionConfig, **kwargs):
        super().__init__(db_config)
        self.db_config = db_config
        self.kwargs = kwargs

    def get_connection(self) -> pymysql.Connection:
        """
        Establishes a connection to the MySQL database using the provided configuration.

        Returns
        -------
            pymysql.connections.Connection: The connection object for the MySQL database.

        Raises
        ------
            None

        Notes: - If SSL/TLS is enabled and configured in the RDSConnectionConfig object,
        the connection will be established with SSL/TLS. - If SSL/TLS is not enabled or configured,
        the connection will be established without SSL/TLS.

        """
        params = dict(
            host=self.db_config.endpoint,
            port=self.db_config.port,
            user=self.db_config.database_username,
            password=self.get_password(),
            db=self.db_config.database_name,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",
            connect_timeout=self.db_config.database_connection_timeout,
        )
        if self.db_config.ssl and self.db_config.ssl_enabled:
            return pymysql.connect(
                ssl=self.db_config.ssl.__dict__,
                **params,
            )

        return pymysql.connect(
            **params,
        )

    def get_db_cursor(self, connection) -> Cursor:
        """
        Returns a cursor object for executing queries on the database.

        Args:
            - connection (pymysql.connections.Connection): The connection object for the
            MySQL database.

        Returns
        -------
            pymysql.cursors.Cursor: The cursor object for executing queries on the database.

        Notes
        -----
            - The cursor object is used to execute queries on the database.
            - The connection object must be established before calling this method.

        """
        return connection.cursor()

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Executes the given query on the database and returns the result as a pandas DataFrame.

        Args:
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
        connection.close()
        _end_time = time.perf_counter() - _start_time
        _LOGGER.info("RDS MYSQL Query: %s, execution time:  %.4fs", query, _end_time)
        return df
