import pandas as pd
import pymysql.cursors
import os
import logging
from numalogic.connectors.aws.boto3_client_manager import Boto3ClientManager
from numalogic.connectors.aws.db_configurations import (
    load_db_conf,
    DatabaseServiceProvider,
    DatabaseTypes,
)
from numalogic.connectors.aws.exceptions import UnRecognizedDatabaseTypeException, \
    UnRecognizedDatabaseServiceProviderException

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


class DBConnector:
    """
    This code snippet defines a class called DBConnector, which is used to connect to a database and execute queries.
    The class has the following methods:

    - __init__(self, db_config, **kwargs): Initializes the DBConnector object with the provided database
    configuration and additional keyword arguments.

    - get_mysql_connection(self, db_config): Establishes a MySQL database connection using the provided database
    configuration.

    - connect(self): Connects to the database based on the specified database provider (RDS or Athena) and returns
    the connection object.

    - execute_query(self, query): Executes the given query on the connected database and returns the result as a
    pandas DataFrame.

    The code also includes an if __name__ == "__main__" block, which demonstrates the usage of the DBConnector class
    by loading a database configuration, creating a DBConnector object, and executing a sample query.

    """

    def __init__(self, db_config, **kwargs):
        self.kwargs = kwargs
        self.db_config = db_config
        self.conn = None
        self.database_type = db_config.database_type

    def get_mysql_connection(self, db_config):
        """
        Establishes a MySQL database connection using the provided database configuration.

        Parameters:
            db_config (object): The database configuration object containing the necessary connection details.

        Returns:
            object: The connection object to the MySQL database.

        Notes:
            - If SSL is enabled in the database configuration, the connection will be established with SSL.
            - The connection uses the pymysql library and the DictCursor cursor class.
            - The connection character set is set to 'utf8mb4'.
            - The connection timeout is set to the value specified in the database configuration.

        """
        os.environ["LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN"] = "1"
        if db_config.ssl and db_config.ssl_enabled:
            self.conn = pymysql.connect(
                host=db_config.endpoint,
                port=db_config.port,
                user=db_config.database_username,
                password=db_config.database_password,
                db=db_config.database_name,
                ssl=db_config.ssl,
                cursorclass=pymysql.cursors.DictCursor,
                charset="utf8mb4",
                connect_timeout=db_config.database_connection_timeout,
            )
            return self.conn
        else:
            self.conn = pymysql.connect(
                host=db_config.endpoint,
                port=db_config.port,
                user=db_config.database_username,
                password=db_config.database_password,
                db=db_config.database_name,
                cursorclass=pymysql.cursors.DictCursor,
                charset="utf8mb4",
                connect_timeout=db_config.database_connection_timeout,
            )
            return self.conn

    def connect(self):
        """
        Connects to the database based on the specified database provider (RDS or Athena) and returns the connection object.

        Returns:
            object: The connection object to the database.

        Raises:
            UnRecognizedDatabaseTypeException: If the specified database type is not supported.
            UnRecognizedDatabaseServiceProviderException: If the specified database service provider is not supported

        Notes:
            - If the database provider is RDS and the 'aws_rds_use_iam' flag is set to True or the database password is empty,
              the method uses AWS IAM authentication to connect to the database.
            - If the database provider is RDS and the 'aws_rds_use_iam' flag is set to False and the database password is not
              empty, the method uses password-based authentication to connect to the database.
            - If the database provider is Athena, the method uses the Boto3ClientManager to get the Athena client and returns it.
        """
        if self.db_config.database_provider.lower() == DatabaseServiceProvider.rds.value:
            db_password = None
            if self.db_config.aws_rds_use_iam or db_password == "":
                print("using aws_rds_use_iam ")
                boto3_client_manager = Boto3ClientManager(self.db_config)
                rds_client = boto3_client_manager.get_client(DatabaseServiceProvider.rds.value)
                db_password = boto3_client_manager.get_rds_token(rds_client)
            else:
                print("using password")
                db_password = self.db_config.database_password

            self.db_config.database_password = db_password

            if self.db_config.database_type.lower() == DatabaseTypes.mysql.value:
                return self.get_mysql_connection(self.db_config)
            else:
                raise UnRecognizedDatabaseTypeException(
                    f"database_type : {self.db_config.database_type.lower()} is not yet supported, please choose one "
                    f"from {DatabaseTypes.list()}"
                )
        elif self.db_config.database_provider.lower() == DatabaseServiceProvider.athena.value:
            boto3_client_manager = Boto3ClientManager(self.db_config)
            athena_client = boto3_client_manager.get_client(
                client_type=DatabaseServiceProvider.athena.value
            )
            return athena_client
        else:
            raise UnRecognizedDatabaseServiceProviderException(
                f"database_provider: {self.db_config.database_provider.lower()} is not yet supported, please choose one"
                f"from {DatabaseServiceProvider.list()}"
            )

    def get_db_cursor(self):
        self.conn = self.connect()
        cursor = self.conn.cursor()
        return cursor

    def execute_query(self, query):
        """
        Executes the given query on the connected database and returns the result as a pandas DataFrame.

        Parameters:
            query (str): The SQL query to be executed on the database.

        Returns:
            pandas.DataFrame: The result of the query as a pandas DataFrame, with column names and rows.

        Raises:
            None

        Notes:
            - This method fetches column names and results for Athena and Mysql/Aurora Mysql databases.
            - For Athena, the method does not perform any specific action.
            - For Mysql/Aurora Mysql, the method establishes a connection to the database, executes the query using a cursor,
              fetches the column names and rows, and returns the result as a pandas DataFrame.

        """
        # Fetching column names and results for Athena
        if self.db_config.database_provider.lower() == DatabaseServiceProvider.athena.value:
            """To Be Implemented for Athena"""
        # Fetching column names and results for Mysql / Aurora Mysql
        elif self.db_config.database_provider.lower() == DatabaseServiceProvider.rds.value:
            """"""
            cursor = self.get_db_cursor()
            cursor.execute(query)
            col_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            res=pd.DataFrame(rows, columns=col_names)
            return pd.DataFrame(rows, columns=col_names)
        else:
            raise UnRecognizedDatabaseServiceProviderException(
                f"database_provider: {self.db_config.database_provider.lower()} is not yet supported, please choose one"
                f"from {DatabaseServiceProvider.list()}"
            )

# if __name__ == "__main__":
#     pd.options.display.max_columns = 1000
#     pd.options.display.max_rows = 1000
#
#     config = load_db_conf(
#         "/Users/skondakindi/Desktop/codebase/odl/odl-ml-python-sdk/tests/resources/db_config_no_ssl.yaml"
#     )
#     _LOGGER.info(config)
#     db_connector = DBConnector(config)
#
#     # result = db_connector.execute_query("""show create table ml_poc.fci_ml_poc5""")
#     result = db_connector.execute_query(
#         """select *  from ml_poc.fci_ml_poc5
#     where   hash_assetid_pluginassetid_iname='3a82e4cf00949ff6ddbf3fdd9ea4dfad' """
#     )
#     _LOGGER.info(result.to_json(orient="records"))
