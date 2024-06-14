import pymysql
import pytest
from unittest.mock import Mock, patch, MagicMock
from nlconnectors.utils.aws.config import (
    DatabaseTypes,
    RDSConnectionConfig as RDSConfig,
)
from nlconnectors.rds.db.mysql_fetcher import MysqlFetcher


@pytest.fixture
def mock_db_config():
    db_config = RDSConfig(
        endpoint="localhost",
        port=3306,
        database_username="username",
        database_password="password",
        database_name="db_name",
        database_connection_timeout=10,
        ssl_enabled=True,
    )

    mock_ssl = Mock()
    mock_ssl.__dict__ = {"foo": "bar"}  # This is just a sample dictionary for testing
    db_config.ssl = mock_ssl

    return db_config, {
        "host": db_config.endpoint,
        "port": db_config.port,
        "user": db_config.database_username,
        "password": db_config.database_password,
        "db": db_config.database_name,
        "cursorclass": pymysql.cursors.DictCursor,
        "charset": "utf8mb4",
        "connect_timeout": db_config.database_connection_timeout,
    }


@pytest.fixture
def mock_db_config_ssl_disabled():
    db_config = RDSConfig(
        endpoint="localhost",
        port=3306,
        database_username="username",
        database_password="password",
        database_name="db_name",
        database_connection_timeout=10,
        ssl_enabled=False,
    )
    return db_config, {
        "host": db_config.endpoint,
        "port": db_config.port,
        "user": db_config.database_username,
        "password": db_config.database_password,
        "db": db_config.database_name,
        "cursorclass": pymysql.cursors.DictCursor,
        "charset": "utf8mb4",
        "connect_timeout": db_config.database_connection_timeout,
    }


@pytest.fixture
def setup_fetcher():
    rds_config = Mock()  # Assuming that your RDSConfig class behaves like a normal python object
    kwargs = {"key": "value"}
    return MysqlFetcher(rds_config, **kwargs)


@pytest.fixture
def mock_mysql_fetcher(mock_db_config):
    db_config, params = mock_db_config
    return MysqlFetcher(db_config=db_config)


@pytest.fixture
def mock_mysql_fetcher_ssl_disabled(mock_db_config_ssl_disabled):
    db_config, params = mock_db_config_ssl_disabled
    return MysqlFetcher(db_config=db_config)


def test_init_method(mock_mysql_fetcher, mock_db_config):
    db_config, params = mock_db_config
    assert mock_mysql_fetcher.db_config == db_config
    assert mock_mysql_fetcher.database_type == DatabaseTypes.MYSQL


def test_get_db_cursor_method(mock_mysql_fetcher):
    mock_connection = Mock()
    mock_cursor = Mock()
    mock_connection.cursor.return_value = mock_cursor
    result = mock_mysql_fetcher.get_db_cursor(mock_connection)
    assert result == mock_cursor


@patch.object(MysqlFetcher, "get_connection")
@patch.object(MysqlFetcher, "get_password")
@patch.object(MysqlFetcher, "get_db_cursor")
def test_execute_query(mock_get_db_cursor, mock_get_password, mock_get_connection):
    # Set mock values
    mock_config = Mock()
    fetcher = MysqlFetcher(mock_config)
    mock_query = "SELECT * FROM MockTable"
    mock_cursor = MagicMock()
    mock_get_db_cursor.return_value = mock_cursor

    mock_cursor.description = [("col1",), ("col2",)]
    mock_cursor.fetchall.return_value = [("val1", "val2")]

    # Execute the method
    result = fetcher.execute_query(mock_query)

    # Check if the methods were called in sequence
    mock_get_connection.assert_called_once()
    mock_get_db_cursor.assert_called_once_with(mock_get_connection.return_value)
    mock_cursor.execute.assert_called_once_with(mock_query)

    # It should return a dataframe
    assert result.to_dict() == {"col1": {0: "val1"}, "col2": {0: "val2"}}


@patch("pymysql.connect", return_value="connection")
def test_get_connection_method(mock_mysql_connect, mock_mysql_fetcher, mock_db_config):
    db_config, params = mock_db_config
    result = mock_mysql_fetcher.get_connection()
    assert result == "connection"

    if mock_mysql_fetcher.db_config.ssl and mock_mysql_fetcher.db_config.ssl_enabled:
        mock_mysql_connect.assert_called_once_with(
            ssl=mock_mysql_fetcher.db_config.ssl.__dict__, **params
        )
    else:
        mock_mysql_connect.assert_called_once_with(**params)


@patch("pymysql.connect", return_value="connection")
def test_get_connection_method_no_ssl(
    mock_mysql_connect, mock_mysql_fetcher_ssl_disabled, mock_db_config_ssl_disabled
):
    db_config, params = mock_db_config_ssl_disabled
    result = mock_mysql_fetcher_ssl_disabled.get_connection()
    assert result == "connection"

    if (
        mock_mysql_fetcher_ssl_disabled.db_config.ssl
        and mock_mysql_fetcher_ssl_disabled.db_config.ssl_enabled
    ):
        mock_mysql_connect.assert_called_once_with(**params)
    else:
        mock_mysql_connect.assert_called_once_with(**params)
