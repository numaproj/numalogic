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
        ds_config = self.cm.get_ds_config(config_name="sandbox_numalogic_demo1")
        self.assertTrue(ds_config)
        self.assertEqual(ds_config.name, "sandbox_numalogic_demo1")

        # from given default config
        ds_config = self.cm.get_ds_config(config_name="argo-rollouts")
        self.assertTrue(ds_config)
        self.assertEqual(ds_config.name, "argo-rollouts")

        # default config
        ds_config = self.cm.get_ds_config(config_name="random")
        self.assertTrue(ds_config)
        self.assertEqual(ds_config.name, "default")

    def test_get_metric_config(self):
        # from given config
        metric_config = self.cm.get_metric_config(
            config_name="sandbox_numalogic_demo1", metric_name="rollout_latency"
        )
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "rollout_latency")

        # from given default config
        metric_config = self.cm.get_metric_config(
            config_name="argo-rollouts",
            metric_name="namespace_app_rollouts_http_request_error_rate",
        )
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "namespace_app_rollouts_http_request_error_rate")

        # from given default config
        metric_config = self.cm.get_metric_config(
            config_name="fciAsset",
            metric_name="ciStatus_failed",
        )
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "ciStatus_failed")

        # default config
        metric_config = self.cm.get_metric_config(config_name="random", metric_name="random_metric")
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "random_metric")
        self.assertTrue(metric_config.numalogic_conf)

    def test_get_unified_config(self):
        # from given config
        unified_config = self.cm.get_unified_config(config_name="sandbox_numalogic_demo1")
        self.assertTrue(unified_config)

        # from given default config
        unified_config = self.cm.get_unified_config(config_name="default-argorollouts")
        self.assertTrue(unified_config)

        # default config - will not have unified config
        unified_config = self.cm.get_unified_config(config_name="random")
        self.assertTrue(unified_config.strategy, "max")

    def test_get_datastream_config_time(self):
        _start_time = time.perf_counter()
        ConfigManager.get_ds_config(config_name="sandbox_numalogic_demo1")
        time1 = time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        ConfigManager.get_ds_config(config_name="sandbox_numalogic_demo1")
        time2 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        self.assertTrue(time2 <= time1)

    def test_get_metric_config_time(self):
        _start_time = time.perf_counter()
        ConfigManager().get_metric_config(
            config_name="sandbox_numalogic_demo1", metric_name="rollout_latency"
        )
        time1 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        ConfigManager().get_metric_config(
            config_name="sandbox_numalogic_demo1", metric_name="rollout_latency"
        )
        time2 = time.perf_counter() - _start_time
        self.assertTrue(time2 < time1)

    def test_get_unified_config_time(self):
        _start_time = time.perf_counter()
        ConfigManager().get_unified_config(config_name="sandbox_numalogic_demo1")
        time1 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        ConfigManager().get_unified_config(config_name="sandbox_numalogic_demo1")
        time2 = time.perf_counter() - _start_time
        self.assertTrue(time2 < time1)
