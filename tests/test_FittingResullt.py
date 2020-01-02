import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())
from models.FittingResult import *


class TestComponentFittingResult(unittest.TestCase):
    def setUp(self):
        self.real_x = np.linspace(-10, 10, 2001)
        self.fitting_space_x = np.array([i+1 for i in self.real_x])
        self.params = np.array([0, 2], dtype=np.float64)
        self.fraction = 1.0
        self.algorithm_data = AlgorithmData(DistributionType.Normal, 1)

    def tearDown(self):
        self.real_x = None
        self.fitting_space_x = None
        self.params = None
        self.fraction = None
        self.algorithm_data = None

    def gen_by_defaluts(self):
        component_result = ComponentFittingResult(
            self.real_x, self.fitting_space_x,
            self.algorithm_data,
            self.params, self.fraction)
        return component_result

    def test_valid(self):
        component_result = self.gen_by_defaluts()
        self.assertFalse(component_result.has_nan)

    def test_has_attrs(self):
        component_result = self.gen_by_defaluts()
        component_result.component_y
        component_result.params
        component_result.fraction
        component_result.mean
        component_result.median
        component_result.mode
        component_result.variance
        component_result.standard_deviation
        component_result.skewness
        component_result.kurtosis

    def test_read_only(self):
        component_result = self.gen_by_defaluts()
        with self.assertRaises(AttributeError):
            component_result.component_y = None
        with self.assertRaises(AttributeError):
            component_result.params = None
        with self.assertRaises(AttributeError):
            component_result.fraction = 0.0
        with self.assertRaises(AttributeError):
            component_result.mean = 0.0
        with self.assertRaises(AttributeError):
            component_result.median = 0.0
        with self.assertRaises(AttributeError):
            component_result.mode = 0.0
        with self.assertRaises(AttributeError):
            component_result.variance = 0.0
        with self.assertRaises(AttributeError):
            component_result.standard_deviation = 0.0
        with self.assertRaises(AttributeError):
            component_result.skewness = 0.0
        with self.assertRaises(AttributeError):
            component_result.kurtosis = 0.0

    def test_fraction_invalid(self):
        self.fraction = -1.0
        with self.assertRaises(AssertionError):
            component_result = self.gen_by_defaluts()
        self.fraction = None
        with self.assertRaises(AssertionError):
            component_result = self.gen_by_defaluts()
        self.fraction = "0.5"
        with self.assertRaises(AssertionError):
            component_result = self.gen_by_defaluts()

    def test_param_invalid(self):
        self.params[0] = np.nan
        component_result = self.gen_by_defaluts()

        self.assertTrue(component_result.has_nan)
        self.assertTrue(np.isnan(component_result.fraction))
        self.assertTrue(np.all(np.isnan(component_result.component_y)))
        self.assertTrue(np.isnan(component_result.mean))
        self.assertTrue(np.isnan(component_result.median))
        self.assertTrue(np.isnan(component_result.mode))
        self.assertTrue(np.isnan(component_result.variance))
        self.assertTrue(np.isnan(component_result.standard_deviation))
        self.assertTrue(np.isnan(component_result.skewness))
        self.assertTrue(np.isnan(component_result.kurtosis))

    def test_param_out_of_range(self):
        self.params[0] = -20
        component_result = self.gen_by_defaluts()
        self.assertTrue(component_result.has_nan)
        self.assertTrue(np.isnan(component_result.mean))
        self.assertTrue(np.isnan(component_result.median))
        self.assertTrue(np.isnan(component_result.mode))


class TestFittingResult(unittest.TestCase):
    def setUp(self):
        self.name = "Test Sample"
        self.real_x = np.linspace(0.01, 10, 1001)
        self.bin_numbers = np.array([i+1 for i in range(len(self.real_x))])
        self.fitting_space_x = self.bin_numbers
        self.algorithm_data = AlgorithmData(DistributionType.Weibull, 3)
        self.fitted_params = np.array([2.41, 2.56, 4.21, 4.50, 2.54, 6.32, 0.23, 0.53])
        self.target_y = self.algorithm_data.mixed_func(self.fitting_space_x, *self.fitted_params)
        self.x_offset = 0.0

    def tearDown(self):
        self.name = None
        self.real_x = None
        self.bin_numbers = None
        self.fitting_space_x = None
        self.algorithm_data = None
        self.fitted_params = None
        self.target_y = None
        self.x_offset = None

    def gen_by_defaluts(self):
        fitting_result = FittingResult(
            self.name, self.real_x, self.fitting_space_x,
            self.bin_numbers, self.target_y, self.algorithm_data,
            self.fitted_params, self.x_offset)
        return fitting_result

    def test_valid(self):
        fitting_result = self.gen_by_defaluts()
        self.assertFalse(fitting_result.has_invalid_value)
        self.assertEqual(len(fitting_result.components), 3)
        self.assertAlmostEqual(fitting_result.mean_squared_error, 0.0)

    def test_has_attrs(self):
        fitting_result = self.gen_by_defaluts()
        fitting_result.uuid
        fitting_result.name
        fitting_result.real_x
        fitting_result.fitting_space_x
        fitting_result.bin_numbers
        fitting_result.target_y
        fitting_result.distribution_type
        fitting_result.component_number
        fitting_result.param_count
        fitting_result.param_names
        fitting_result.components
        fitting_result.fitted_y
        fitting_result.mean_squared_error
        fitting_result.pearson_r
        fitting_result.kendall_tau
        fitting_result.spearman_r
        fitting_result.has_invalid_value

    def test_read_only(self):
        fitting_result = self.gen_by_defaluts()
        with self.assertRaises(AttributeError):
            fitting_result.uuid = None
        with self.assertRaises(AttributeError):
            fitting_result.name = None
        with self.assertRaises(AttributeError):
            fitting_result.real_x = None
        with self.assertRaises(AttributeError):
            fitting_result.fitting_space_x = None
        with self.assertRaises(AttributeError):
            fitting_result.bin_numbers = None
        with self.assertRaises(AttributeError):
            fitting_result.target_y = None
        with self.assertRaises(AttributeError):
            fitting_result.distribution_type = None
        with self.assertRaises(AttributeError):
            fitting_result.component_number = None
        with self.assertRaises(AttributeError):
            fitting_result.param_count = None
        with self.assertRaises(AttributeError):
            fitting_result.param_names = None
        with self.assertRaises(AttributeError):
            fitting_result.components = None
        with self.assertRaises(AttributeError):
            fitting_result.fitted_y = None
        with self.assertRaises(AttributeError):
            fitting_result.mean_squared_error = None
        with self.assertRaises(AttributeError):
            fitting_result.pearson_r = None
        with self.assertRaises(AttributeError):
            fitting_result.kendall_tau = None
        with self.assertRaises(AttributeError):
            fitting_result.spearman_r = None
        with self.assertRaises(AttributeError):
            fitting_result.has_invalid_value = None

    def test_correlation_coefficient(self):
        fitting_result = self.gen_by_defaluts()
        coefficient, p = fitting_result.pearson_r
        self.assertAlmostEqual(coefficient, 1.0)
        self.assertAlmostEqual(p, 0.0)
        coefficient, p = fitting_result.kendall_tau
        self.assertAlmostEqual(coefficient, 1.0)
        self.assertAlmostEqual(p, 0.0)
        coefficient, p = fitting_result.spearman_r
        self.assertAlmostEqual(coefficient, 1.0)
        self.assertAlmostEqual(p, 0.0)

    def test_param_has_nan(self):
        self.fitted_params[0] = np.nan
        fitting_result = self.gen_by_defaluts()
        self.assertTrue(fitting_result.has_invalid_value)

        for comp in fitting_result.components[1:]:
            self.assertFalse(comp.has_nan)

    def test_update(self):
        fitting_result = self.gen_by_defaluts()
        y1 = fitting_result.fitted_y
        self.fitted_params[0] = 4.2145
        fitting_result.update(self.fitted_params, self.x_offset)
        y2 = fitting_result.fitted_y
        self.assertIsNot(y1, y2)


if __name__ == "__main__":
    unittest.main()
