import pytest
from unittest.mock import patch
from pandas import DataFrame, to_datetime

from numalogic.connectors._config import Pivot
from numalogic.connectors.rds._base import RDSBase, RDSConfig, Boto3ClientManager


@pytest.fixture(autouse=True)
def mock_db_config():
    mock_db_config = RDSConfig()
    mock_db_config.database_type = "db_type"
    return mock_db_config


@pytest.fixture(autouse=True)
def data_fetcher(mock_db_config):
    data_fetcher = RDSBase(db_config=mock_db_config)
    return data_fetcher


@pytest.fixture(autouse=True)
def mock_boto3(mocker):
    return mocker.MagicMock(spec=Boto3ClientManager)


# @pytest.fixture(autouse=True)
# def setUp( mocker):
#     mock_boto3 = mocker.MagicMock(spec=Boto3ClientManager)
#     mock_db_config = mocker.MagicMock(spec=RDSConfig)
#     mock_db_config.database_type = "db_type"
#     data_fetcher = RDSBase(db_config=mock_db_config)

def test_init(mock_db_config, data_fetcher):
    assert data_fetcher.db_config == mock_db_config
    assert data_fetcher.database_type is not None


def test_get_password_else(data_fetcher, mock_db_config):
    mock_db_config.aws_rds_use_iam = False
    mock_db_config.database_password = "password"
    result = data_fetcher.get_password()
    assert result == "password"


def test_get_password_aws(mocker, data_fetcher, mock_db_config):
    mocker.patch.object(RDSBase, "get_rds_token", return_value="password")
    mock_db_config.aws_rds_use_iam = True
    result = data_fetcher.get_password()
    assert result == "password"


def test_get_rds_token(data_fetcher, mocker):
    mocker.patch.object(
        Boto3ClientManager,
        "get_client"
    )

    mocker.patch.object(
        Boto3ClientManager,
        "get_rds_token",
        return_value="password"
    )
    result = data_fetcher.get_rds_token()
    assert result == "password"


def test_format_dataframe(data_fetcher):
    df = DataFrame({"eventdatetime": ["2020-01-01 00:00:00"]})
    expected_df = DataFrame({"eventdatetime": ["2020-01-01 00:00:00"]})
    query = "SELECT * FROM table"
    datetime_field_name = "eventdatetime"
    df = data_fetcher.format_dataframe(df, query, datetime_field_name)
    print(df.to_json())
    assert (
            df["timestamp"] == to_datetime(expected_df["eventdatetime"]).astype("int64") // 10 ** 6
    ).all()


def test_format_dataframe_without_group_by_pivot(data_fetcher):
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
    df = data_fetcher.format_dataframe(df, query, datetime_field_name)
    expected = [
        {"asset": "123", "count": 2, "timestamp": 1708527420000},
        {"asset": "123", "count": 3, "timestamp": 1708527420000},
        {"asset": "123", "count": 8, "timestamp": 1708527420000},
    ]
    assert expected == df.to_dict(orient="records")


def test_format_dataframe_group_by(data_fetcher):
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
    df = data_fetcher.format_dataframe(df, query, datetime_field_name, group_by=group_by)
    expected = [{"asset": "123", "count": 13, "timestamp": 5125582260000}]
    assert expected == df.to_dict(orient="records")


def test_format_dataframe_pivot(data_fetcher):
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
    df = data_fetcher.format_dataframe(
        df, query, datetime_field_name, group_by=group_by, pivot=pivot
    )
    expected = [{"degraded": 8, "failed": 8, "success": 5, "timestamp": 1708527420000}]
    assert df.to_dict(orient="records") == expected


def test_execute_query(mocker, data_fetcher):
    mocker.patch.object(RDSBase, "execute_query", return_value=DataFrame())
    dataframe_result = data_fetcher.execute_query("SELECT * FROM table")
    assert isinstance(dataframe_result, DataFrame)


def test_get_db_cursor(data_fetcher):
    assert data_fetcher.get_db_cursor() is None


def test_execute_query_exception(data_fetcher):
    with pytest.raises(NotImplementedError):
        data_fetcher.execute_query("SELECT * FROM table")


def test_get_connection(data_fetcher):
    with pytest.raises(NotImplementedError):
        data_fetcher.get_connection()


def test_raises_error_if_database_not_available(mocker, data_fetcher):
    # Mock the necessary dependencies
    db_config = RDSConfig(
        database_type="mysql", aws_rds_use_iam=False, database_password="password"
    )
    data_fetcher = RDSBase(db_config)
    mocker.patch.object(
        data_fetcher, "get_connection", side_effect=Exception("RDS database not available")
    )

    # Invoke the method under test and assert that it raises an error
    with pytest.raises(Exception):
        data_fetcher.get_connection()
