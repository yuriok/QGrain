import unittest
import os
import sys

sys.path.append(os.getcwd())

from algorithms import check_ncomp
class TestCheckNcomp(unittest.TestCase):
    def test_check_ncomp(self):
        # There is at least one component, ncomp` should be int and greater than 0.
        with self.assertRaises(TypeError):
            check_ncomp("1")
        with self.assertRaises(TypeError):
            check_ncomp(1.4)
        with self.assertRaises(TypeError):
            check_ncomp([1])
        with self.assertRaises(ValueError):
            check_ncomp(0)
        with self.assertRaises(ValueError):
            check_ncomp(-1)

from algorithms import get_params
import numpy as np
class TestGetParams(unittest.TestCase):
    def test_count(self):
        i = np.random.randint(1, 100)
        params = get_params(i)
        self.assertEqual(len(params), i*3-1)

    def test_location(self):
        i = np.random.randint(1, 100)
        params = get_params(i)

        check_set = set()
        for param in params:
            self.assertIsInstance(param["location"], int)
            self.assertGreaterEqual(param["location"], 0)
            self.assertLess(param["location"], len(params))
            check_set.add(param["location"])

        # check if there is any duplicate location using the feature of `set`
        self.assertEqual(len(check_set), len(params))


from algorithms import get_mixed_weibull
from algorithms import weibull, double_weibull, triple_weibull, quadruple_weibull
class TestGetMixedWeibull(unittest.TestCase):
    def test_callable(self):
        func = get_mixed_weibull(3)
    
    def test_ncomp1_validity(self):
        x = np.linspace(0.01, 10, 1000)
        params = [2.7, 2.4]
        expected_y = weibull(x, *params)
        generated_func = get_mixed_weibull(1)
        real_y = generated_func(x, *params)
        self.assertTrue(all(expected_y==real_y))

    def test_ncomp2_validity(self):
        x = np.linspace(0.01, 10, 1000)
        params = [2.1, 4.1, 1.5, 5.2, 0.4]
        expected_y = double_weibull(x, *params)
        generated_func = get_mixed_weibull(2)
        real_y = generated_func(x, *params)
        self.assertTrue(all(expected_y==real_y))

    def test_ncomp3_validity(self):
        x = np.linspace(0.01, 10, 1000)
        params = [2.1, 2.4, 4.1, 1.5, 2.5, 5.2, 0.4, 0.1]
        expected_y = triple_weibull(x, *params)
        generated_func = get_mixed_weibull(3)
        real_y = generated_func(x, *params)
        self.assertTrue(all(expected_y==real_y))

    def test_ncomp4_validity(self):
        x = np.linspace(0.01, 10, 1000)
        params = [2.1, 2.4, 4.1, 1.5, 2.5, 5.2, 3.1, 1.6, 0.3, 0.1, 0.2]
        expected_y = quadruple_weibull(x, *params, )
        generated_func = get_mixed_weibull(4)
        real_y = generated_func(x, *params)
        self.assertTrue(all(expected_y==real_y))


if __name__ == "__main__":
    unittest.main()
