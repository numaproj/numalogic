import unittest

from anomalydetection.factory import HandlerFactory

from anomalydetection.udf import window


class TestFactory(unittest.TestCase):
    def test_preprocess(self):
        func = HandlerFactory.get_handler("window")
        self.assertEqual(func, window)

    def test_invalid(self):
        with self.assertRaises(NotImplementedError):
            HandlerFactory.get_handler("Lionel Messi")


if __name__ == "__main__":
    unittest.main()
