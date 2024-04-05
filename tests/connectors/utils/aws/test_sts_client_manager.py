import unittest
from unittest.mock import patch, MagicMock
from numalogic.connectors.utils.aws.sts_client_manager import STSClientManager
from datetime import datetime, timedelta, timezone
class TestSTSClientManager(unittest.TestCase):
    @patch('numalogic.connectors.utils.aws.sts_client_manager.boto3.client')
    def test_STSClientManager(self, boto3_client_mock):
        # Prepare the mock methods
        mock_sts_client = MagicMock()
        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "test_key",
                "SecretAccessKey": "test_access_key",
                "SessionToken": "test_token",
                "Expiration": (datetime.now(timezone.utc) + timedelta(hours=1))
            }
        }
        boto3_client_mock.return_value = mock_sts_client

        manager = STSClientManager()

        # Test assume_role
        role_arn = "test_arn"
        role_session_name = "test_session"
        manager.assume_role(role_arn, role_session_name)
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn=role_arn,
            RoleSessionName=role_session_name,
            DurationSeconds=3600
        )
        self.assertEqual(manager.credentials, mock_sts_client.assume_role.return_value["Credentials"])

        # Test is_token_about_to_expire
        self.assertFalse(manager.is_token_about_to_expire())

        # Test get_credentials
        credentials = manager.get_credentials(role_arn, role_session_name)
        self.assertEqual(manager.credentials, mock_sts_client.assume_role.return_value["Credentials"])
        self.assertEqual(credentials, mock_sts_client.assume_role.return_value["Credentials"])

        # Test renew of credentials
        manager.credentials["Expiration"] = datetime.now(timezone.utc)
        credentials = manager.get_credentials(role_arn, role_session_name)
        self.assertEqual(credentials, mock_sts_client.assume_role.return_value["Credentials"])

if __name__ == '__main__':
    unittest.main()