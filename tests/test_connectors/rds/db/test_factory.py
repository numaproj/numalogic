import pytest
from connectors import RdsFactory
from connectors import MysqlFetcher
from connectors import UnRecognizedDatabaseTypeException


def test_get_db_handler_with_supported_db_type():
    # Arrange
    db_type = "mysql"
    # Act
    result = RdsFactory.get_db_handler(db_type)

    # Assert
    assert result == MysqlFetcher


def test_get_db_handler_with_unsupported_db_type():
    # Arrange
    db_type = "not_supported"

    # Act and Assert
    with pytest.raises(UnRecognizedDatabaseTypeException):
        RdsFactory.get_db_handler(db_type)
