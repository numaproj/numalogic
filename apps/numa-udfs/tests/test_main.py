import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from pynumaflow.mapper import MapServer, MapMultiprocServer

from src.constants import TESTS_DIR
from numalogic.tools.exceptions import ConfigNotFoundError

BASE_CONFIG_PATH = f"{TESTS_DIR}/resources/_config3.yaml"
APP_CONFIG_PATH = f"{TESTS_DIR}/resources/_config4.yaml"
REDIS_AUTH = "123"


class TestMainScript(unittest.TestCase):
    @patch.dict("os.environ", {"BASE_CONF_PATH": BASE_CONFIG_PATH, "REDIS_AUTH": REDIS_AUTH})
    def test_init_server_01(self):
        from src.__main__ import init_server

        server = init_server("preprocess", "sync")
        self.assertIsInstance(server, MapServer)

    @patch.dict("os.environ", {"BASE_CONF_PATH": BASE_CONFIG_PATH, "REDIS_AUTH": REDIS_AUTH})
    def test_init_server_02(self):
        from src.__main__ import init_server

        server = init_server("inference", "multiproc")
        self.assertIsInstance(server, MapMultiprocServer)

    def test_conf_loader(self):
        from src import load_pipeline_conf

        plconf = load_pipeline_conf(BASE_CONFIG_PATH, APP_CONFIG_PATH)
        base_conf = OmegaConf.load(BASE_CONFIG_PATH)
        app_conf = OmegaConf.load(APP_CONFIG_PATH)

        self.assertListEqual(
            list(plconf.stream_confs),
            list(base_conf["stream_confs"]) + list(app_conf["stream_confs"]),
        )

    def test_conf_loader_appconf_not_exist(self):
        from src import load_pipeline_conf

        app_conf_path = "_random.yaml"
        plconf = load_pipeline_conf(BASE_CONFIG_PATH, app_conf_path)
        base_conf = OmegaConf.load(BASE_CONFIG_PATH)

        self.assertListEqual(list(plconf.stream_confs), list(base_conf["stream_confs"]))

    def test_conf_loader_err(self):
        from src import load_pipeline_conf

        with self.assertRaises(ConfigNotFoundError):
            load_pipeline_conf("_random1.yaml", "_random2.yaml")


if __name__ == "__main__":
    unittest.main()
