import logging

from connectors import DatabaseTypes
from connectors import UnRecognizedDatabaseTypeException

_LOGGER = logging.getLogger(__name__)


class RdsFactory:
    """class represents a factory for creating database handlers for different database types."""

    @classmethod
    def get_db_handler(cls, database_type: DatabaseTypes):
        """
        Get the database handler for the specified database type.

        Args:
            - database_type (str): The type of the database.

        Returns
        -------
            - The database handler for the specified database type.

        Raises
        ------
            - UnRecognizedDatabaseTypeException: If the specified database type is not supported.

        """
        if database_type == DatabaseTypes.MYSQL:
            from connectors import MysqlFetcher

            return MysqlFetcher

        raise UnRecognizedDatabaseTypeException(f"database_type: {database_type} is not supported")
