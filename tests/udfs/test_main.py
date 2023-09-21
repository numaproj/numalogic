import unittest
from unittest.mock import patch

from pynumaflow.mapper import Mapper, MultiProcMapper

from numalogic._constants import TESTS_DIR

CONFIG_PATH = f"{TESTS_DIR}/udfs/resources/_config.yaml"
REDIS_AUTH = "123"


class TestMainScript(unittest.TestCase):
    @patch.dict("os.environ", {"CONF_PATH": CONFIG_PATH, "REDIS_AUTH": REDIS_AUTH})
    def test_init_server_01(self):
        from numalogic.udfs.__main__ import init_server

        server = init_server("preprocess", "sync")
        self.assertIsInstance(server, Mapper)

    @patch.dict("os.environ", {"CONF_PATH": CONFIG_PATH, "REDIS_AUTH": REDIS_AUTH})
    def test_init_server_02(self):
        from numalogic.udfs.__main__ import init_server

        server = init_server("inference", "multiproc")
        self.assertIsInstance(server, MultiProcMapper)


if __name__ == "__main__":
    unittest.main()
