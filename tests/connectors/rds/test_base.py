import pytest
from unittest.mock import patch
from pandas import DataFrame
from numalogic.connectors.rds._base import RDSDataFetcher, RDSConfig, Boto3ClientManager


class TestRDSDataFetcher:
    @pytest.fixture(autouse=True)
    def setUp(self, mocker):
        self.mock_boto3 = mocker.MagicMock(spec=Boto3ClientManager)
        self.mock_rds_config = mocker.MagicMock(spec=RDSConfig)
        self.mock_rds_config.database_type = "db_type"
        self.data_fetcher = RDSDataFetcher(db_config=self.mock_rds_config)

    def test_init(self):
        assert self.data_fetcher.db_config == self.mock_rds_config
        assert self.data_fetcher.database_type is not None

    def test_get_password_else(self):
        self.mock_rds_config.aws_rds_use_iam = False
        self.mock_rds_config.database_password = "password"
        result = self.data_fetcher.get_password()
        assert result == "password"

    @pytest.mark.parametrize("mock_get_rds_token, expected_result", [("password", "password")])
    def test_get_password_aws(self, mocker, mock_get_rds_token, expected_result):
        mocker.patch.object(RDSDataFetcher, "get_rds_token", return_value=mock_get_rds_token)
        self.mock_rds_config.aws_rds_use_iam = True
        result = self.data_fetcher.get_password()
        assert result == expected_result

    @patch.object(Boto3ClientManager, "get_client", autospec=True)
    @patch.object(Boto3ClientManager, "get_rds_token", autospec=True)
    def test_get_rds_token(self, mock_get_rds_token, mock_get_client):
        mock_get_rds_token.return_value = "password"
        result = self.data_fetcher.get_rds_token()
        assert result == "password"

    def test_execute_query(self, mocker):
        mocker.patch.object(RDSDataFetcher, "execute_query", return_value=DataFrame())
        dataframe_result = self.data_fetcher.execute_query("SELECT * FROM table")
        assert isinstance(dataframe_result, DataFrame)
