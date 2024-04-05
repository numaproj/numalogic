import unittest
from unittest.mock import Mock, patch
from numalogic.connectors.rds._rds import RDSFetcher


class TestRDSFetcher(unittest.TestCase):

    @patch('numalogic.connectors.rds._rds.db')
    def test_init(self, mock_db):
        config = Mock()
        fetcher = RDSFetcher(config)

        self.assertEqual(fetcher.db_config, config)
        # If a CLASS_TYPE is defined, it should have been instantiated
        # with the db_config.
        if mock_db.CLASS_TYPE:
            # db.CLASS_TYPE constructor should have been called with db_config
            mock_db.CLASS_TYPE.assert_called_once_with(config)
            # fetcher.fetcher should be an instance of db.CLASS_TYPE
            self.assertIsInstance(fetcher.fetcher,
                                  mock_db.CLASS_TYPE.return_value.__class__)

    @patch('numalogic.connectors.rds._rds.db')
    def test_fetch(self, mock_db):
        config = Mock()
        fetcher = RDSFetcher(config)
        mock_db.CLASS_TYPE.return_value.execute_query.return_value = "Expected Result"
        result = fetcher.fetch("select 1")
        self.assertEqual(result, "Expected Result")
        # It should call db.CLASS_TYPE.execute_query with query
        mock_db.CLASS_TYPE.return_value.execute_query.assert_called_once_with("select 1")


if __name__ == "__main__":
    unittest.main()
