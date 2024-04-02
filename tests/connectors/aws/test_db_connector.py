import unittest
from unittest.mock import patch, Mock, MagicMock
import pymysql
from numalogic.connectors.aws.exceptions import UnRecognizedDatabaseTypeException, \
    UnRecognizedDatabaseServiceProviderException
from numalogic.connectors.aws.db_connector import DBConnector
from numalogic.connectors.aws.boto3_client_manager import Boto3ClientManager
from numalogic.connectors.aws.db_configurations import DatabaseServiceProvider, DatabaseTypes


# replace your_module with the actual name of your module

class TestDBConnector(unittest.TestCase):

    @patch.object(pymysql, 'connect')
    def test_get_mysql_connection(self, mock_mysql_connect):
        mock_db_config = Mock()
        mock_db_config.ssl = False
        mock_db_config.ssl_enabled = False
        db_connector = DBConnector(mock_db_config)

        db_connector.get_mysql_connection(mock_db_config)

        mock_mysql_connect.assert_called_once()

    @patch.object(pymysql, 'connect')
    def test_get_mysql_connection_ssl(self, mock_mysql_connect):
        mock_db_config = Mock()
        mock_db_config.ssl = True
        mock_db_config.ssl_enabled = True
        db_connector = DBConnector(mock_db_config)

        db_connector.get_mysql_connection(mock_db_config)

        mock_mysql_connect.assert_called_once()

    def test_connect_unknown_database(self):
        mock_db_config = Mock()
        mock_db_config.database_type.lower.return_value = 'unknown'
        db_connector = DBConnector(mock_db_config)

        with self.assertRaises(UnRecognizedDatabaseServiceProviderException):
            db_connector.connect()

    @patch.object(DBConnector, 'get_mysql_connection')
    @patch.object(Boto3ClientManager, 'get_rds_token')
    @patch.object(Boto3ClientManager, 'get_client')
    def test_connect_rds_mysql(self, mock_get_mysql_connection, mock_get_rds_token, mock_get_client):
        mock_db_config = Mock()
        mock_db_config.database_provider.lower.return_value = DatabaseServiceProvider.rds.value
        mock_db_config.database_type.lower.return_value = "mysql"
        db_connector = DBConnector(mock_db_config)

        db_connector.connect()

        mock_get_mysql_connection.assert_called_once()


    @patch.object(DBConnector,'connect')
    @patch.object(Boto3ClientManager, 'get_client')
    @patch.object(Boto3ClientManager, 'get_rds_token')
    def test_execute_query(self, mock_connect, get_client_mock, get_rds_token_mock):
        # Prepare the mock methods
        get_rds_token_mock.return_value = 'dummy_rds_token'
        get_client_mock.return_value = 'dummy_client'
        mock_connect.return_value = 'dummy_db_connection'

        mock_config = MagicMock()
        mock_config.database_provider = DatabaseServiceProvider.rds.value
        mock_config.aws_rds_use_iam = True
        mock_config.database_password = ''
        mock_config.database_type = 'mysql'

        connector = DBConnector(mock_config)

        # Mock pymysql's DictCursor object
        dummy_cursor = MagicMock()
        dummy_cursor.description = [(None, None, None, None, None, None, None), (None, None, None, None, None, None, None)]
        dummy_cursor.fetchall.return_value = [{},{}]

        # Mock the get_db_cursor method in DBConnector
        with patch.object(connector, 'get_db_cursor', return_value=dummy_cursor) as mock_get_db_cursor:
            df = connector.execute_query("SELECT * FROM table")
            mock_get_db_cursor.assert_called_once()


    def test_connect_unknown_databaseserviceprovider(self):
        mock_config = MagicMock()
        mock_config.database_provider = "unkown"
        mock_config.aws_rds_use_iam = True
        mock_config.database_password = ''
        mock_config.database_type = 'mysql'

        db_connector = DBConnector(mock_config)

        with self.assertRaises(UnRecognizedDatabaseServiceProviderException):
            db_connector.connect()


    def test_connect_unknown_databasetype(self):
        mock_config = MagicMock()
        mock_config.database_provider = DatabaseServiceProvider.rds.value
        mock_config.aws_rds_use_iam = False
        mock_config.database_password = ''
        mock_config.database_type = 'unknown_db'

        db_connector = DBConnector(mock_config)

        with self.assertRaises(UnRecognizedDatabaseTypeException):
            db_connector.connect()
    @patch.object(DBConnector, 'get_mysql_connection')
    def test_use_database_password(self, mock_get_mysql_connection):
        mock_config = MagicMock()
        mock_config.database_provider = DatabaseServiceProvider.rds.value
        mock_config.aws_rds_use_iam = False
        mock_config.database_type = 'mysql'
        mock_config.database_password = "test"
        db_connector = DBConnector(mock_config)
        db_connector.connect()
        self.assertEqual("test", db_connector.db_config.database_password)



if __name__ == "__main__":
    unittest.main()
