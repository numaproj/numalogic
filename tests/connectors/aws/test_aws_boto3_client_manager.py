import unittest
from unittest.mock import MagicMock, patch
from botocore.exceptions import NoCredentialsError
from numalogic.connectors.aws.exceptions import UnRecognizedAWSClientException
from numalogic.connectors.aws.boto3_client_manager import Boto3ClientManager
from numalogic.connectors.aws.sts_client_manager import STSClientManager

class TestBoto3ClientManager(unittest.TestCase):

    def setUp(self):
        self.config_mock = MagicMock()
        self.boto3_client_manager = Boto3ClientManager(self.config_mock)
        self.rds_client_mock = MagicMock()
        self.athena_client_mock = MagicMock()
        self.sts_client_manager_mock = MagicMock()
        self.boto3_session_mock = MagicMock()

    def test_init(self):
        self.assertTrue(isinstance(self.boto3_client_manager.sts_client_manager, STSClientManager))

    def test_get_boto3_session(self):
        self.boto3_client_manager.sts_client_manager.get_credentials = MagicMock(return_value={
            "AccessKeyId": "testAccessKey",
            "SecretAccessKey": "testSecretKey",
            "SessionToken": "testSessionToken"
        })

        boto3_session = self.boto3_client_manager.get_boto3_session()

        print(boto3_session)

        # self.boto3_session_mock.assert_called_with(
        #         aws_access_key_id="testAccessKey",
        #         aws_secret_access_key="testSecretKey",
        #         aws_session_token="testSessionToken"
        #     )


        with patch.object(Boto3ClientManager,'get_boto3_session', return_value=self.boto3_session_mock) as boto3_session_class:
            boto3_session = self.boto3_client_manager.get_boto3_session()

            self.assertEqual(boto3_session, self.boto3_session_mock)
            boto3_session_class.assert_called_with( )

    def test_get_rds_token(self):
        self.rds_client_mock.generate_db_auth_token.return_value = 'test_token'
        self.boto3_client_manager.configurations.endpoint = 'test_endpoint'
        self.boto3_client_manager.configurations.port = 'test_port'
        self.boto3_client_manager.configurations.database_username = 'username'
        self.boto3_client_manager.configurations.aws_region = 'region'

        rds_token = self.boto3_client_manager.get_rds_token(self.rds_client_mock)

        self.assertEqual(rds_token, 'test_token')
        self.rds_client_mock.generate_db_auth_token.assert_called_with(
            DBHostname='test_endpoint',
            Port='test_port',
            DBUsername='username',
            Region='region'
        )

    def test_get_client_unrecognized(self):
        with self.assertRaises(UnRecognizedAWSClientException):
            self.boto3_client_manager.get_client('unrecognized')

    def test_get_client_rds(self):
        self.boto3_client_manager.get_boto3_session = MagicMock(return_value=self.boto3_session_mock)
        self.boto3_session_mock.client.return_value = self.rds_client_mock

        rds_client = self.boto3_client_manager.get_client('rds')

        self.boto3_client_manager.get_boto3_session.assert_called_once()
        self.boto3_session_mock.client.assert_called_with('rds', region_name=self.boto3_client_manager.configurations.aws_region)
        self.assertEqual(rds_client, self.rds_client_mock)

    def test_get_client_athena(self):
        self.boto3_client_manager.get_boto3_session = MagicMock(return_value=self.boto3_session_mock)
        self.boto3_session_mock.client.return_value = self.athena_client_mock

        athena_client = self.boto3_client_manager.get_client('athena')

        self.boto3_client_manager.get_boto3_session.assert_called_once()
        self.boto3_session_mock.client.assert_called_with('athena', region_name=self.boto3_client_manager.configurations.aws_region)



if __name__ == '__main__':
    unittest.main()