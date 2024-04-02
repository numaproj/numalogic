import unittest
from unittest.mock import patch, Mock
import numalogic.connectors.aws.db_configurations as db_configuration  # Replace with the actual module name
from numalogic.connectors.aws.exceptions import ConfigNotFoundError
from numalogic._constants import TESTS_DIR
import os


class TestAwsConfig(unittest.TestCase):
    def test_load_db_conf_file_exists(self):
        result = db_configuration.load_db_conf(os.path.join(TESTS_DIR, "resources", "rds_db_config.yaml"))
        # here add the asserts to validate that the config has been created as expected
        self.assertIsNotNone(result)

    @patch("numalogic.connectors.aws.db_configurations.OmegaConf.load", side_effect=FileNotFoundError())
    def test_load_db_conf_file_not_exists(self, mock_load):
        path = "/path/doesnotexist/config.yaml"
        with self.assertRaises(ConfigNotFoundError):
            db_configuration.load_db_conf(path)

    def test_RDSConfig_defaults(self):
        config = db_configuration.RDSConfig()
        self.assertEqual(config.aws_region, "")
        self.assertFalse(config.aws_rds_use_iam)


if __name__ == "__main__":
    unittest.main()
