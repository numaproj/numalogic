from unittest.mock import patch

import pytest
from pandas import DataFrame

from numalogic.connectors.rds._config import RDSConfig
from numalogic.connectors.rds._rds import RDSFetcher
import pandas as pd


class TestRDSFetcher:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.mock_db_config = RDSConfig()
        self.data_fetcher = RDSFetcher(db_config=self.mock_db_config)

    def test_init(self):
        assert self.data_fetcher.db_config == self.mock_db_config

    def test_fetch(self, mocker):
        mock_data = pd.DataFrame(data={"col1": ["value1", "value2"], "col2": ["value3", "value4"]})

        mocker.patch.object(RDSFetcher, "fetch", return_value=mock_data)

        result = self.data_fetcher.fetch("SELECT * FROM table")

        assert isinstance(result, pd.DataFrame)

    def test_execute_query(self, mocker):
        rds_config = RDSConfig(database_type="mysql")
        rds_fetcher = RDSFetcher(db_config=rds_config)
        mocker.patch.object(rds_fetcher.fetcher, "execute_query", return_value=DataFrame())
        result = rds_fetcher.fetch("SELECT * FROM table", datetime_field_name="eventdatetime")
        assert result.empty == DataFrame().empty

    def test_rds_fetcher_fetch(self):
        rds_config = RDSConfig(database_type="mysql")
        rds_fetcher = RDSFetcher(db_config=rds_config)
        with patch.object(rds_fetcher.fetcher, "execute_query") as mock_query:
            mock_query.return_value = pd.DataFrame({"test": [1, 2, 3]})
            result = rds_fetcher.fetch("SELECT * FROM test", "test")
            mock_query.assert_called_once_with("SELECT * FROM test")
            assert not result.empty

    def test_raw_fetch(self):
        with pytest.raises(NotImplementedError):
            self.data_fetcher.raw_fetch()
