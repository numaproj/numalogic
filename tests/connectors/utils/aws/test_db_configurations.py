from unittest.mock import patch
import pytest
import numalogic.connectors.utils.aws.db_configurations as db_configuration
from numalogic._constants import TESTS_DIR
import os


class TestAwsConfig:
    def test_load_db_conf_file_exists(self):
        result = db_configuration.load_db_conf(
            os.path.join(TESTS_DIR, "resources", "rds_db_config.yaml")
        )
        assert result is not None

    @patch(
        "numalogic.connectors.utils.aws.db_configurations.OmegaConf.load",
        side_effect=FileNotFoundError(),
    )
    def test_load_db_conf_file_not_exists(self, mock_load):
        path = "/path/doesnotexist/config.yaml"
        with pytest.raises(db_configuration.ConfigNotFoundError):
            db_configuration.load_db_conf(path)

    def test_RDSConfig_defaults(self):
        config = db_configuration.RDSConfig()
        assert config.aws_region == ""
        assert config.aws_rds_use_iam is False
