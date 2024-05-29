from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

from nlconnectors.utils.aws.sts_client_manager import STSClientManager


@patch("nlconnectors.utils.aws.sts_client_manager.boto3.client")
def test_sts_client_manager(boto3_client_mock):
    # Prepare the mock methods
    mock_sts_client = MagicMock()
    mock_sts_client.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "test_key",
            "SecretAccessKey": "test_access_key",
            "SessionToken": "test_token",
            "Expiration": (datetime.now(timezone.utc) + timedelta(hours=1)),
        }
    }
    boto3_client_mock.return_value = mock_sts_client

    manager = STSClientManager()

    # Test assume_role
    role_arn = "test_arn"
    role_session_name = "test_session"
    manager.assume_role(role_arn, role_session_name)
    mock_sts_client.assume_role.assert_called_once_with(
        RoleArn=role_arn, RoleSessionName=role_session_name, DurationSeconds=3600
    )
    assert (
        manager.credentials == mock_sts_client.assume_role.return_value["Credentials"]
    )

    # Test is_token_about_to_expire
    assert manager.is_token_about_to_expire() is False

    # Test get_credentials
    credentials = manager.get_credentials(role_arn, role_session_name)
    assert (
        manager.credentials == mock_sts_client.assume_role.return_value["Credentials"]
    )
    assert credentials == mock_sts_client.assume_role.return_value["Credentials"]

    # Test renew of credentials
    manager.credentials["Expiration"] = datetime.now(timezone.utc)
    credentials = manager.get_credentials(role_arn, role_session_name)
    assert credentials == mock_sts_client.assume_role.return_value["Credentials"]
