import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from numalogic.preprocess.transformer import LogTransformer, StaticPowerTransformer


class TestTransformers(unittest.TestCase):
    def test_logtransformer(self):
        x = 1 + np.random.randn(5, 3)
        transformer = LogTransformer(add_factor=1)
        x_prime = transformer.transform(x)

        assert_almost_equal(np.log1p(x), x_prime)
        assert_almost_equal(transformer.fit_transform(x), x_prime)
        assert_almost_equal(transformer.inverse_transform(x_prime), np.expm1(x_prime))

    def test_staticpowertransformer(self):
        x = 1 + np.random.randn(5, 3)
        transformer = StaticPowerTransformer(3, add_factor=2)
        x_prime = transformer.transform(x)

        assert_almost_equal(np.power(2 + x, 3), x_prime)
        assert_almost_equal(transformer.fit_transform(x), x_prime)
        assert_almost_equal(transformer.inverse_transform(x_prime), x, decimal=4)


if __name__ == "__main__":
    unittest.main()
