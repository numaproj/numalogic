import os
import socket
import unittest
import numpy as np
from unittest.mock import patch, Mock

from src.tools import is_host_reachable, WindowScorer
from src.watcher import ConfigManager
from tests import mock_configs


def mock_resolver(*_, **__):
    raise socket.gaierror


class TestTools(unittest.TestCase):
    INFER_OUT = None

    def test_is_host_reachable(self):
        self.assertTrue(is_host_reachable("google.com"))

    @patch("src.tools.get_ipv4_by_hostname", mock_resolver)
    def test_is_host_reachable_err(self):
        self.assertFalse(is_host_reachable("google.com", max_retries=2, sleep_sec=1))


@patch.object(ConfigManager, "load_configs", Mock(return_value=mock_configs()))
class TestWindowScorer(unittest.TestCase):
    def test_get_winscore(self):
        static_threshold = ConfigManager().get_static_threshold_config(
            config_name="sandbox_numalogic_demo2",
            metric_name="namespace_app_rollouts_http_request_error_rate",
        )
        postprocess_conf = ConfigManager().get_postprocess_config(
            config_name="sandbox_numalogic_demo2",
            metric_name="namespace_app_rollouts_http_request_error_rate",
        )

        stream = np.random.uniform(low=1, high=2, size=(10, 1))
        winscorer = WindowScorer(static_threshold, postprocess_conf)
        final_score = winscorer.get_ensemble_score(stream)
        self.assertLess(final_score, 3.0)
