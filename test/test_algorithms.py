import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())
from algorithms import *


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
            check_ncomp(1)
        with self.assertRaises(ValueError):
            check_ncomp(-1)

class TestGetParams(unittest.TestCase):
    def test_count_weibull(self):
        for i in range(2, 100):
            params = get_params(i, DistributionType.Weibull)
            self.assertEqual(len(params), i*3-1)

    def test_location_weibull(self):
        for i in range(2, 100):
            params = get_params(i, DistributionType.Weibull)

            check_set = set()
            for param in params:
                self.assertIsInstance(param["location"], int)
                self.assertGreaterEqual(param["location"], 0)
                self.assertLess(param["location"], len(params))
                check_set.add(param["location"])

            # check if there is any duplicate location using the feature of `set`
            self.assertEqual(len(check_set), len(params))


class TestGetMixedWeibull(unittest.TestCase):
    def test_callable(self):
        func = get_mixed_weibull(3)
    
    def test_tuple_length(self):
        for i in range(2, 100):
            res = get_mixed_weibull(i)
            self.assertEqual(len(res), len(["func", "bounds", "constrains", "default_values", "func_params"]))

    def test_bounds(self):
        for i in range(2, 100):
            bounds = get_mixed_weibull(i)[1]
            self.assertEqual(len(bounds), i*3-1)
            for bound in bounds:
                self.assertEqual(len(bound), 2)

    def test_defaults(self):
        for i in range(2, 100):
            defaults = get_mixed_weibull(i)[3]
            self.assertEqual(len(defaults), i*3-1)
            for default in defaults:
                self.assertIsNotNone(default)
    
    def test_func_params(self):
        params_keys = ["name", "default", "location", "bounds"]
        for i in range(2, 100):
            func_params = get_mixed_weibull(i)[4]
            self.assertEqual(len(func_params), i*3-1)
            for param in func_params:
                self.assertIsInstance(param, dict)
                for key in params_keys:
                    self.assertTrue(key in param.keys())

    def test_ncomp2_validity(self):
        x = np.linspace(0.01, 10, 1000)
        params = [2.1, 4.1, 1.5, 5.2, 0.4]
        expected_y = double_weibull(x, *params)
        generated_func = get_mixed_weibull(2)[0]
        real_y = generated_func(x, *params)
        self.assertTrue(all(expected_y==real_y))

    def test_ncomp3_validity(self):
        x = np.linspace(0.01, 10, 1000)
        params = [2.1, 2.4, 4.1, 1.5, 2.5, 5.2, 0.4, 0.1]
        expected_y = triple_weibull(x, *params)
        generated_func = get_mixed_weibull(3)[0]
        real_y = generated_func(x, *params)
        self.assertTrue(all(expected_y==real_y))

    def test_ncomp4_validity(self):
        x = np.linspace(0.01, 10, 1000)
        params = [2.1, 2.4, 4.1, 1.5, 2.5, 5.2, 3.1, 1.6, 0.3, 0.1, 0.2]
        expected_y = quadruple_weibull(x, *params, )
        generated_func = get_mixed_weibull(4)[0]
        real_y = generated_func(x, *params)
        self.assertTrue(all(expected_y==real_y))


if __name__ == "__main__":
    unittest.main()
