import unittest
from unittest.mock import patch, Mock
from argparse import Namespace
from pandas import DataFrame

from numalogic.connectors.rds._base import RDSDataFetcher, RDSConfig, UnRecognizedDatabaseTypeException, \
    UnRecognizedDatabaseServiceProviderException, Boto3ClientManager
from numalogic.connectors.rds._config import DatabaseTypes


class TestRDSDataFetcher(unittest.TestCase):

    @patch("numalogic.connectors.utils.aws.boto3_client_manager.Boto3ClientManager")
    def setUp(self, boto3_client_manager_mock):
        self.mock_boto3 = Mock(spec=Boto3ClientManager)
        self.mock_rds_config = Mock(spec=RDSConfig)
        self.mock_rds_config.database_type = "db_type"

        self.data_fetcher = RDSDataFetcher(db_config=self.mock_rds_config)

    def test_init(self):
        self.assertIsNotNone(self.data_fetcher)
        self.assertEqual(self.data_fetcher.db_config, self.mock_rds_config)
        self.assertIsNotNone(self.data_fetcher.database_type)

    def test_get_password_else(self):
        self.mock_rds_config.aws_rds_use_iam = False
        self.mock_rds_config.database_password = "password"
        result = self.data_fetcher.get_password()
        self.assertEqual(result, "password")

    @patch.object(RDSDataFetcher, 'get_rds_token')
    def test_get_password_aws(self, mock_get_rds_token):
        mock_get_rds_token.return_value = "password"
        self.mock_rds_config.aws_rds_use_iam = True
        result = self.data_fetcher.get_password()
        self.assertEqual(result, "password")

    @patch.object(Boto3ClientManager, 'get_client', autospec=True)
    @patch.object(Boto3ClientManager, 'get_rds_token', autospec=True)
    def test_get_rds_token(self, mock_get_rds_token, mock_get_client):
        mock_get_rds_token.return_value = "password"
        result = self.data_fetcher.get_rds_token()
        self.assertEqual(result, "password")

    def test_execute_query(self):
        self.data_fetcher.execute_query = Mock(return_value=DataFrame())
        dataframe_result = self.data_fetcher.execute_query('SELECT * FROM table')
        self.assertIsInstance(dataframe_result, DataFrame)


if __name__ == '__main__':
    unittest.main()
