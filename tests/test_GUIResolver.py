import os
import sys
import unittest

import numpy as np
from scipy.stats import norm

sys.path.append(os.getcwd())
from resolvers.GUIResolver import *


class TestGUIResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = GUIResolver()
        self.resolver.distribution_type = DistributionType.Normal
        self.resolver.component_number = 1
        self.resolver.sigFittingSucceeded.connect(self.on_fitting_finished)
        self.fitting_result = None
        x = np.linspace(-10, 10, 201)
        y = norm.pdf(x, 5.45, 2.21)
        self.default_sample_data = SampleData("Sample", x, y)

    def on_fitting_finished(self, fitting_result: FittingResult):
        self.fitting_result = fitting_result

    def test_valid(self):
        self.resolver.on_target_data_changed(self.default_sample_data)
        self.resolver.try_fit()
        self.assertIsNotNone(self.fitting_result)

    def test_on_component_number_changed(self):
        for i in range(1, 10):
            self.resolver.on_component_number_changed(i)
            self.assertEqual(self.resolver.component_number, i)

    def test_on_distribution_number_changed(self):
        for distribution_type in [DistributionType.Normal,
                       DistributionType.Weibull,
                       DistributionType.GeneralWeibull]:
            self.resolver.on_distribution_type_changed(distribution_type)
            self.assertEqual(self.resolver.distribution_type, distribution_type)

    def test_on_settings_changed(self):
        # valid keys
        self.resolver.on_inherit_params_changed(True)
        self.assertTrue(self.resolver.inherit_params)
        self.resolver.on_inherit_params_changed(False)
        self.assertFalse(self.resolver.inherit_params)
        # invalid keys
        with self.assertRaises(NotImplementedError):
            self.resolver.on_inherit_params_changed({"some_not_exist_key": True})

    def test_on_algorithm_settings_changed(self):
        settings = dict(
            global_optimization_maxiter=100,
            global_optimization_success_iter=5,
            global_optimization_stepsize=2.0,
            minimizer_tolerance=1e-10,
            minimizer_maxiter=500,
            final_tolerance=1e-100,
            final_maxiter=1000)
        self.resolver.on_algorithm_settings_changed(settings)

    def test_data_not_prepared(self):
        self.resolver.try_fit()

    def test_inherit_params(self):
        self.resolver.on_inherit_params_changed({"inherit_params": True})
        self.resolver.on_target_data_changed(self.default_sample_data)
        self.resolver.try_fit()
        self.assertIsNotNone(self.resolver.last_succeeded_params)
        last = self.resolver.last_succeeded_params
        x = np.linspace(-10, 10, 201)
        y = norm.pdf(x, 5.45, 2.21)
        new_sample_data = SampleData("New Sample", x, y)
        self.resolver.on_target_data_changed(new_sample_data)
        self.resolver.try_fit()
        # the last succeeded params were set as the initial guess
        self.assertIs(self.resolver.initial_guess, last)

    def test_not_inherit_params(self):
        self.resolver.on_inherit_params_changed({"inherit_params": False})
        self.resolver.on_target_data_changed(self.default_sample_data)
        self.resolver.try_fit()
        self.assertIsNotNone(self.resolver.last_succeeded_params)
        last = self.resolver.last_succeeded_params
        x = np.linspace(-10, 10, 201)
        y = norm.pdf(x, 5.45, 2.21)
        new_sample_data = SampleData("New Sample", x, y)
        self.resolver.on_target_data_changed(new_sample_data)
        self.resolver.try_fit()
        # the last succeeded params were not set as the initial guess
        self.assertIsNot(self.resolver.initial_guess, last)
        # and the defaults were set as the initial guess
        self.assertIs(self.resolver.initial_guess, self.resolver.algorithm_data.defaults)


if __name__ == "__main__":
    unittest.main()
