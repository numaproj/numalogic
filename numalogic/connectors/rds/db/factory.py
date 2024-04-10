import logging

from numalogic.connectors.utils.aws.exceptions import UnRecognizedDatabaseTypeException

_LOGGER = logging.getLogger(__name__)


class RdsFactory:
    """
    Class: RdsFactory.

    This class represents a factory for creating database handlers for different database types.

    Methods
    -------
    - get_db_handler(database_type: str) -> Type[DatabaseHandler]:
        - This method takes a database_type as input and returns
            the corresponding database handler class.
        - If the database_type is "mysql", it returns the MysqlFetcher class from
            the numalogic.connectors.rds.db.mysql_fetcher module.
        - If the database_type is not supported, it returns None.

    """

    @classmethod
    def get_db_handler(cls, database_type: str):
        db_class = None
        if database_type == "mysql":
            from numalogic.connectors.rds.db.mysql_fetcher import MysqlFetcher

            db_class = MysqlFetcher
        else:
            raise UnRecognizedDatabaseTypeException(
                f"database_type: {database_type} is not supported"
            )
        return db_class
