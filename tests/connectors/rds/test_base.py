import pytest
from unittest.mock import patch
from pandas import DataFrame, to_datetime

from numalogic.connectors._config import Pivot
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

    def test_format_dataframe(self):
        df = DataFrame({"eventdatetime": ["2020-01-01 00:00:00"]})
        expected_df = DataFrame({"eventdatetime": ["2020-01-01 00:00:00"]})
        query = "SELECT * FROM table"
        datetime_field_name = "eventdatetime"
        df = self.data_fetcher.format_dataframe(df, query, datetime_field_name)
        print(df.to_json())
        assert (
            df["timestamp"] == to_datetime(expected_df["eventdatetime"]).astype("int64") // 10**6
        ).all()

    def test_format_dataframe_without_group_by_pivot(self):
        df = DataFrame.from_dict(
            [
                {"eventdatetime": "2024-02-21T14:57:00Z", "asset": "123", "count": 2},
                {"eventdatetime": "2024-02-21T14:57:00Z", "asset": "123", "count": 3},
                {"eventdatetime": "2024-02-21T14:57:00Z", "asset": "123", "count": 8},
            ],
            orient="columns",
        )

        query = "SELECT * FROM table"
        datetime_field_name = "eventdatetime"
        df = self.data_fetcher.format_dataframe(df, query, datetime_field_name)
        expected = [
            {"asset": "123", "count": 2, "timestamp": 1708527420000},
            {"asset": "123", "count": 3, "timestamp": 1708527420000},
            {"asset": "123", "count": 8, "timestamp": 1708527420000},
        ]
        assert expected == df.to_dict(orient="records")

    def test_format_dataframe_group_by(self):
        df = DataFrame.from_dict(
            [
                {"eventdatetime": "2024-02-21T14:57:00Z", "asset": "123", "count": 2},
                {"eventdatetime": "2024-02-21T14:57:00Z", "asset": "123", "count": 3},
                {"eventdatetime": "2024-02-21T14:57:00Z", "asset": "123", "count": 8},
            ],
            orient="columns",
        )

        query = "SELECT * FROM table"
        datetime_field_name = "eventdatetime"
        group_by = ["asset"]
        df = self.data_fetcher.format_dataframe(df, query, datetime_field_name, group_by=group_by)
        expected = [{"asset": "123", "count": 13, "timestamp": 5125582260000}]
        assert expected == df.to_dict(orient="records")

    def test_format_dataframe_pivot(self):
        df = DataFrame.from_dict(
            [
                {
                    "eventdatetime": "2024-02-21T14:57:00Z",
                    "asset": "123",
                    "count": 2,
                    "ciStatus": "success",
                },
                {
                    "eventdatetime": "2024-02-21T14:57:00Z",
                    "asset": "123",
                    "count": 3,
                    "ciStatus": "success",
                },
                {
                    "eventdatetime": "2024-02-21T14:57:00Z",
                    "asset": "123",
                    "count": 8,
                    "ciStatus": "failed",
                },
                {
                    "eventdatetime": "2024-02-21T14:57:00Z",
                    "asset": "123",
                    "count": 8,
                    "ciStatus": "degraded",
                },
            ],
            orient="columns",
        )
        query = "SELECT * FROM table"
        datetime_field_name = "eventdatetime"
        group_by = ["timestamp", "ciStatus"]
        pivot = Pivot()
        pivot.columns = ["ciStatus"]
        df = self.data_fetcher.format_dataframe(
            df, query, datetime_field_name, group_by=group_by, pivot=pivot
        )
        expected = [{"degraded": 8, "failed": 8, "success": 5, "timestamp": 1708527420000}]
        assert df.to_dict(orient="records") == expected

    def test_execute_query(self, mocker):
        mocker.patch.object(RDSDataFetcher, "execute_query", return_value=DataFrame())
        dataframe_result = self.data_fetcher.execute_query("SELECT * FROM table")
        assert isinstance(dataframe_result, DataFrame)

    def test_get_db_cursor(self):
        assert self.data_fetcher.get_db_cursor() is None

    def test_execute_query_exception(self):
        with pytest.raises(NotImplementedError):
            self.data_fetcher.execute_query("SELECT * FROM table")

    def test_get_connection(self):
        with pytest.raises(NotImplementedError):
            self.data_fetcher.get_connection()

    def test_raises_error_if_database_not_available(self, mocker):
        # Mock the necessary dependencies
        db_config = RDSConfig(
            database_type="mysql", aws_rds_use_iam=False, database_password="password"
        )
        data_fetcher = RDSDataFetcher(db_config)
        mocker.patch.object(
            data_fetcher, "get_connection", side_effect=Exception("RDS database not available")
        )

        # Invoke the method under test and assert that it raises an error
        with pytest.raises(Exception):
            data_fetcher.get_connection()
