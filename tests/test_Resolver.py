import types
import os
import sys
import unittest

import numpy as np
from scipy.stats import norm

sys.path.append(os.getcwd())
from resolvers.Resolver import *


exec_flag = False
class TestResolver(unittest.TestCase):
    def override(self, *args, **kwargs):
        global exec_flag
        assert not exec_flag
        exec_flag = True

    def setUp(self):
        self.resolver = Resolver()
        self.resolver.component_number = 1
        self.resolver.distribution_type = DistributionType.Normal
        self.x = np.linspace(0.1, 10, 100)
        self.y = norm.pdf(self.x, 2.53, 3.71)

    def tearDown(self):
        global exec_flag
        exec_flag = False

    def test_distribution_type(self):
        self.resolver.distribution_type = DistributionType.Normal
        self.assertEqual(self.resolver.distribution_type, DistributionType.Normal)
        self.resolver.distribution_type = DistributionType.Weibull
        self.assertEqual(self.resolver.distribution_type, DistributionType.Weibull)
        self.resolver.distribution_type = DistributionType.GeneralWeibull
        self.assertEqual(self.resolver.distribution_type, DistributionType.GeneralWeibull)

    def test_component_number(self):
        self.resolver.component_number = 21
        self.assertEqual(self.resolver.component_number, 21)
        self.resolver.component_number = 43
        self.assertEqual(self.resolver.component_number, 43)
        self.resolver.component_number = 56
        self.assertEqual(self.resolver.component_number, 56)
        self.resolver.component_number = 91
        self.assertEqual(self.resolver.component_number, 91)

    def test_get_valid_data_range(self):
        raw_data_list = [
            [0, 1, 2, 4, 5, 0],
            [1, 2, 4, 7, 6, 0],
            [0, 1, 2, 4, 5],
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0],
            [0, 0, 0, 1, 2, 3],
            [0, 0, 1, 2, 4, 0, 0, 0, 1, 2, 0]]
        expected_data_list = [
            [0, 1, 2, 4, 5, 0],
            [1, 2, 4, 7, 6, 0],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 3, 4, 5, 6, 0],
            [0, 1, 2, 3],
            [0, 1, 2, 4, 0, 0, 0, 1, 2, 0]]
        for raw, expected in zip(raw_data_list, expected_data_list):
            left, right = self.resolver.get_valid_data_range(raw)
            actual = raw[left: right]
            self.assertListEqual(actual, expected)

    def test_has_hooks(self):
        hooks = ["on_data_fed",
                 "on_data_not_prepared",
                 "on_fitting_started",
                 "on_fitting_finished",
                 "on_global_fitting_failed",
                 "on_global_fitting_succeeded",
                 "on_final_fitting_failed",
                 "on_exception_raised_while_fitting",
                 "on_fitting_succeeded",
                 "local_iteration_callback",
                 "global_iteration_callback"]

        attrs = dir(self.resolver)
        for hook_name in hooks:
            self.assertTrue(hook_name in attrs)

    def test_feed_data(self):
        self.resolver.feed_data(SampleData("Sample_021", self.x, self.y))
        self.assertEqual(self.resolver.sample_name, "Sample_021")
        self.assertIs(self.resolver.real_x, self.x)
        self.assertIs(self.resolver.target_y, self.y)

    # test hooks
    def test_on_data_fed(self):
        self.resolver.on_data_fed = types.MethodType(self.override, self.resolver)
        self.resolver.feed_data(SampleData("test_on_data_fed", self.x, self.y))
        self.assertTrue(exec_flag)

    def test_on_data_not_prepare(self):
        self.resolver.on_data_not_prepared = types.MethodType(self.override, self.resolver)
        self.resolver.try_fit()
        self.assertTrue(exec_flag)

    def test_on_fitting_started(self):
        self.resolver.on_fitting_started = types.MethodType(self.override, self.resolver)
        self.resolver.feed_data(SampleData("test_on_fitting_started", self.x, self.y))
        self.resolver.try_fit()
        self.assertTrue(exec_flag)

    def test_on_fitting_finished(self):
        self.resolver.on_fitting_finished = types.MethodType(self.override, self.resolver)
        self.resolver.feed_data(SampleData("test_on_fitting_finished", self.x, self.y))
        self.resolver.try_fit()
        self.assertTrue(exec_flag)

    def test_on_fitting_succeeded(self):
        self.resolver.on_fitting_succeeded = types.MethodType(self.override, self.resolver)
        self.resolver.feed_data(SampleData("test_on_fitting_succeeded", self.x, self.y))
        self.resolver.try_fit()
        self.assertTrue(exec_flag)

    def test_change_settings(self):
        settings = dict(
                global_optimization_maxiter=100,
                global_optimization_success_iter=5,
                global_optimization_stepsize=2.0,
                minimizer_tolerance=1e-10,
                minimizer_maxiter=500,
                final_tolerance=1e-100,
                final_maxiter=1000)
        self.resolver.change_settings(**settings)


if __name__ == "__main__":
    unittest.main()
