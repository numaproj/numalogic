import time
import unittest
from unittest.mock import patch, Mock

from src.watcher import ConfigManager
from tests import mock_configs


@patch.object(ConfigManager, "load_configs", Mock(return_value=mock_configs()))
class TestConfigManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cm = ConfigManager

    def test_update_configs(self):
        config = self.cm.update_configs()
        self.assertTrue(len(config), 3)

    def test_load_configs(self):
        app_configs, default_configs, default_numalogic, pipeline_config = self.cm.load_configs()
        self.assertTrue(app_configs)
        self.assertTrue(default_configs)
        self.assertTrue(default_numalogic)
        self.assertTrue(pipeline_config)

    def test_get_datastream_config(self):
        # from users config
        stream_conf = self.cm.get_stream_config(config_id="app1-config")
        self.assertTrue(stream_conf)
        self.assertEqual(stream_conf.config_id, "app1-config")

        # from given default config
        stream_conf = self.cm.get_stream_config(config_id="druid-config")
        self.assertTrue(stream_conf)
        self.assertEqual(stream_conf.config_id, "druid-config")

        # default config
        stream_conf = self.cm.get_stream_config(config_id="random")
        self.assertTrue(stream_conf)
        self.assertEqual(stream_conf.config_id, "default")

    def test_get_unified_config(self):
        # from given user config
        unified_config = self.cm.get_unified_config(config_id="app1-config")
        self.assertTrue(unified_config)

        # from given default config
        unified_config = self.cm.get_unified_config(config_id="prometheus-config")
        self.assertTrue(unified_config)

        # default config - will not have unified config
        unified_config = self.cm.get_unified_config(config_id="random")
        self.assertTrue(unified_config.strategy, "max")

    def test_get_datastream_config_time(self):
        _start_time = time.perf_counter()
        ConfigManager.get_stream_config(config_id="druid-config")
        time1 = time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        ConfigManager.get_stream_config(config_id="druid-config")
        time2 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        self.assertTrue(time2 <= time1)

    def test_get_unified_config_time(self):
        _start_time = time.perf_counter()
        ConfigManager().get_unified_config(config_id="druid-config")
        time1 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        ConfigManager().get_unified_config(config_id="druid-config")
        time2 = time.perf_counter() - _start_time
        self.assertTrue(time2 < time1)
