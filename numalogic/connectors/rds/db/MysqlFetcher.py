from numalogic.connectors.rds._base import RDSDataFetcher
import os
import pymysql
import pandas as pd
import logging

from numalogic.connectors.rds._config import DatabaseTypes, RDSConfig
from numalogic.connectors.utils.aws.db_configurations import load_db_conf

_LOGGER = logging.getLogger(__name__)


class MysqlFetcher(RDSDataFetcher):
    database_type = DatabaseTypes.mysql.value

    def __init__(self, db_config: RDSConfig, **kwargs):
        super().__init__(db_config)
        self.db_config = db_config
        self.kwargs = kwargs

    def get_connection(self):
        # os.environ["LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN"] = "1"
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
        cursor = connection.cursor()
        return cursor

    def execute_query(self, query) -> pd.DataFrame:
        connection = self.get_connection()
        cursor = self.get_db_cursor(connection)
        cursor.execute(query)
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=col_names)


