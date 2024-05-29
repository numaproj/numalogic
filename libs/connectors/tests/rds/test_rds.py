from unittest.mock import patch

import pytest
from pandas import DataFrame

from nlconnectors.config import RDSConnectionConfig as RDSConfig
from nlconnectors import RDSFetcher
import pandas as pd


@pytest.fixture(autouse=True)
def mock_db_config():
    return RDSConfig()


@pytest.fixture(autouse=True)
def data_fetcher(mock_db_config):
    return RDSFetcher(db_config=mock_db_config)


def test_init(data_fetcher, mock_db_config):
    assert data_fetcher.db_config == mock_db_config


def test_fetch(mocker, data_fetcher):
    mock_data = pd.DataFrame(
        data={"col1": ["value1", "value2"], "col2": ["value3", "value4"]}
    )

    mocker.patch.object(RDSFetcher, "fetch", return_value=mock_data)

    result = data_fetcher.fetch("SELECT * FROM table")

    assert isinstance(result, pd.DataFrame)


def test_execute_query(mocker):
    rds_config = RDSConfig(database_type="mysql")
    rds_fetcher = RDSFetcher(db_config=rds_config)
    mocker.patch.object(rds_fetcher.fetcher, "execute_query", return_value=DataFrame())
    result = rds_fetcher.fetch(
        "SELECT * FROM table", datetime_column_name="eventdatetime"
    )
    assert result.empty == DataFrame().empty


def test_rds_fetcher_fetch():
    rds_config = RDSConfig(database_type="mysql")
    rds_fetcher = RDSFetcher(db_config=rds_config)
    with patch.object(rds_fetcher.fetcher, "execute_query") as mock_query:
        mock_query.return_value = pd.DataFrame({"test": [1, 2, 3]})
        result = rds_fetcher.fetch("SELECT * FROM test", "test")
        mock_query.assert_called_once_with("SELECT * FROM test")
        assert not result.empty


def test_raw_fetch(data_fetcher):
    with pytest.raises(NotImplementedError):
        data_fetcher.raw_fetch()
