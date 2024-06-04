import pytest
from nlconnectors.rds._rds import RdsFactory
from nlconnectors.rds.db.mysql_fetcher import MysqlFetcher
from nlconnectors.utils.aws.exceptions import UnRecognizedDatabaseTypeException


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
