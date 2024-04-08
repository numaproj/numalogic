import pytest
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
