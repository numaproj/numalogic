import unittest
from unittest.mock import patch, Mock, MagicMock

from numalogic.connectors.rds._config import RDSConfig
from numalogic.connectors.rds.db.MYSQLFetcher import MYSQLFetcher
from numalogic.connectors.utils.aws.boto3_client_manager import Boto3ClientManager


class TestMYSQLFetcher(unittest.TestCase):

    @patch('numalogic.connectors.rds.db.MYSQLFetcher.pymysql.connect')
    @patch.object(MYSQLFetcher, 'get_password')
    def test_get_connection(self, mock_get_password, mock_connect):
        mock_config = Mock()
        fetcher = MYSQLFetcher(mock_config)
        mock_get_password.return_value = 'mock_password'

        fetcher.get_connection()
        # It should call pymysql.connect with the correct arguments
        mock_connect.assert_called_once()

    @patch('numalogic.connectors.rds.db.MYSQLFetcher.pymysql.connect')
    @patch.object(MYSQLFetcher, 'get_password')
    def test_get_connection_else(self, mock_get_password, mock_connect):
        mock_config = Mock(spec=RDSConfig)
        mock_config.ssl = False
        fetcher = MYSQLFetcher(mock_config)
        mock_get_password.return_value = 'mock_password'

        fetcher.get_connection()
        # It should call pymysql.connect with the correct arguments
        mock_connect.assert_called_once()

    @patch('numalogic.connectors.rds.db.MYSQLFetcher.pymysql.connect')
    def test_get_db_cursor(self, mock_connect):
        mock_config = Mock()
        fetcher = MYSQLFetcher(mock_config)
        mock_connect.cursor.return_value = "Mock Cursor"
        result = fetcher.get_db_cursor(mock_connect)

        # It should call connection.cursor() method and return a cursor
        mock_connect.cursor.assert_called_once()
        self.assertEqual(result, "Mock Cursor")

    @patch.object(MYSQLFetcher, 'get_connection')
    @patch.object(MYSQLFetcher, 'get_password')
    @patch.object(MYSQLFetcher, 'get_db_cursor')
    def test_execute_query(self, mock_get_db_cursor, mock_get_password, mock_get_connection):
        # Set mock values
        mock_config = Mock()
        fetcher = MYSQLFetcher(mock_config)
        mock_query = "SELECT * FROM MockTable"
        mock_cursor = MagicMock()
        mock_get_db_cursor.return_value = mock_cursor

        mock_cursor.description = [('col1',), ('col2',)]
        mock_cursor.fetchall.return_value = [("val1", "val2")]

        # Execute the method
        result = fetcher.execute_query(mock_query)

        # Check if the methods were called in sequence
        mock_get_connection.assert_called_once()
        mock_get_db_cursor.assert_called_once_with(mock_get_connection.return_value)
        mock_cursor.execute.assert_called_once_with(mock_query)

        # It should return a dataframe
        self.assertEqual(result.to_dict(), {'col1': {0: 'val1'}, 'col2': {0: 'val2'}})


if __name__ == '__main__':
    unittest.main()
