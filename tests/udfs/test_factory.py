import unittest

from fakeredis import FakeRedis, FakeServer

from numalogic.udfs import UDFFactory, ServerFactory


class TestUDFFactory(unittest.TestCase):
    def test_get_cls_01(self):
        udf_cls = UDFFactory.get_udf_cls("inference")
        self.assertEqual(udf_cls.__name__, "InferenceUDF")

    def test_get_cls_02(self):
        udf_cls = UDFFactory.get_udf_cls("trainer")
        self.assertEqual(udf_cls.__name__, "DruidTrainerUDF")

    def test_get_cls_err(self):
        with self.assertRaises(ValueError):
            UDFFactory.get_udf_cls("some_udf")

    def test_get_instance(self):
        udf = UDFFactory.get_udf_instance("preprocess", r_client=FakeRedis(server=FakeServer()))
        self.assertIsInstance(udf, UDFFactory.get_udf_cls("preprocess"))


class TestServerFactory(unittest.TestCase):
    def test_get_cls(self):
        server_cls = ServerFactory.get_server_cls("sync")
        self.assertEqual(server_cls.__name__, "Mapper")

    def test_get_cls_err(self):
        with self.assertRaises(ValueError):
            ServerFactory.get_server_cls("some_server")

    def test_get_instance(self):
        server = ServerFactory.get_server_instance("multiproc", handler=lambda x: x)
        self.assertIsInstance(server, ServerFactory.get_server_cls("multiproc"))


if __name__ == "__main__":
    unittest.main()
