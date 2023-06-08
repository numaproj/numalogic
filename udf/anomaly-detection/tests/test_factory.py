import unittest

from src.factory import HandlerFactory


class TestFactory(unittest.TestCase):
    def test_preprocess(self):
        func = HandlerFactory.get_handler("preprocess")
        self.assertTrue(func)

    def test_invalid(self):
        with self.assertRaises(NotImplementedError):
            HandlerFactory.get_handler("Lionel Messi")


if __name__ == "__main__":
    unittest.main()
