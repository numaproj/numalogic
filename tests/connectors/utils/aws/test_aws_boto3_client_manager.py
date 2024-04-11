from unittest.mock import MagicMock, patch
import pytest
from boto3 import Session

from numalogic.connectors.utils.aws.exceptions import UnRecognizedAWSClientException
from numalogic.connectors.utils.aws.boto3_client_manager import Boto3ClientManager
from numalogic.connectors.utils.aws.sts_client_manager import STSClientManager


@pytest.fixture(autouse=True)
def config_mock():
    return MagicMock()


@pytest.fixture(autouse=True)
def boto3_client_manager_mock(config_mock):
    return Boto3ClientManager(config_mock)


@pytest.fixture(autouse=True)
def rds_client_mock():
    return MagicMock()


@pytest.fixture(autouse=True)
def athena_client_mock():
    return MagicMock()


@pytest.fixture(autouse=True)
def sts_client_manager_mock():
    return MagicMock()


@pytest.fixture(autouse=True)
def boto3_session_mock():
    return MagicMock()


def test_init(boto3_client_manager_mock):
    assert isinstance(boto3_client_manager_mock.sts_client_manager, STSClientManager)


def test_get_boto3_session(boto3_client_manager_mock):
    boto3_client_manager_mock.sts_client_manager.get_credentials = MagicMock(
        return_value={
            "AccessKeyId": "testAccessKey",
            "SecretAccessKey": "testSecretKey",
            "SessionToken": "testSessionToken",
        }
    )
    with patch.object(
        Boto3ClientManager, "get_boto3_session", return_value=boto3_session_mock
    ) as boto3_session_class:
        boto3_session = boto3_client_manager_mock.get_boto3_session()

        assert boto3_session == boto3_session_mock
        boto3_session_class.assert_called_with()


@patch.object(STSClientManager, "get_credentials")
def test_valid_get_boto3_session(mock_get_credentials):
    # Mock the STSClientManager.get_credentials method
    mock_get_credentials.return_value = {
        "AccessKeyId": "testAccessKey",
        "SecretAccessKey": "testSecretKey",
        "SessionToken": "testSessionToken",
    }

    # Create a mock configurations object
    configurations_mock = MagicMock()
    configurations_mock.aws_assume_role_arn = "testRoleArn"
    configurations_mock.aws_assume_role_session_name = "testSessionName"

    # Create a Boto3ClientManager object with the mock configurations
    boto3_client_manager = Boto3ClientManager(configurations_mock)

    # Call the get_boto3_session method
    boto3_session = boto3_client_manager.get_boto3_session()

    # Assert that the returned object is an instance of Session
    assert isinstance(boto3_session, Session)


def test_get_rds_token(rds_client_mock, boto3_client_manager_mock):
    rds_client_mock.generate_db_auth_token.return_value = "test_token"
    boto3_client_manager_mock.configurations.endpoint = "test_endpoint"
    boto3_client_manager_mock.configurations.port = "test_port"
    boto3_client_manager_mock.configurations.database_username = "username"
    boto3_client_manager_mock.configurations.aws_region = "region"

    rds_token = boto3_client_manager_mock.get_rds_token(rds_client_mock)

    assert rds_token == "test_token"
    rds_client_mock.generate_db_auth_token.assert_called_with(
        DBHostname="test_endpoint", Port="test_port", DBUsername="username", Region="region"
    )


def test_get_client_unrecognized(boto3_client_manager_mock):
    with pytest.raises(UnRecognizedAWSClientException):
        boto3_client_manager_mock.get_client("unrecognized")


def test_get_client_rds(boto3_client_manager_mock, rds_client_mock, boto3_session_mock):
    boto3_client_manager_mock.get_boto3_session = MagicMock(return_value=boto3_session_mock)
    boto3_session_mock.client.return_value = rds_client_mock

    rds_client = boto3_client_manager_mock.get_client("rds")

    boto3_client_manager_mock.get_boto3_session.assert_called_once()
    boto3_session_mock.client.assert_called_with(
        "rds", region_name=boto3_client_manager_mock.configurations.aws_region
    )
    assert rds_client == rds_client_mock


def test_get_client_athena(boto3_client_manager_mock, athena_client_mock, boto3_session_mock):
    boto3_client_manager_mock.get_boto3_session = MagicMock(return_value=boto3_session_mock)
    boto3_session_mock.client.return_value = athena_client_mock

    boto3_client_manager_mock.get_client("athena")

    boto3_client_manager_mock.get_boto3_session.assert_called_once()
    boto3_session_mock.client.assert_called_with(
        "athena", region_name=boto3_client_manager_mock.configurations.aws_region
    )
