import os
import sys
import unittest

import numpy as np

from QGrain.algorithms import *


NORMAL_PARAM_COUNT = 2
WEIBULL_PARAM_COUNT = 2
GENERAL_WEIBULL_PARAM_COUNT = 3

# the component number must be positive int value
class TestCheckComponentNumber(unittest.TestCase):
    # valid cases
    def test_1_to_100(self):
        for i in range(1, 101):
            check_component_number(i)

    # invalid cases
    def test_str(self):
        with self.assertRaises(TypeError):
            check_component_number("1")

    def test_float(self):
        with self.assertRaises(TypeError):
            check_component_number(1.4)

    def test_list(self):
        with self.assertRaises(TypeError):
            check_component_number([1])

    def test_zero(self):
        with self.assertRaises(ValueError):
            check_component_number(0)

    def test_positive(self):
        with self.assertRaises(ValueError):
            check_component_number(-1)


class TestGetParamCount(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(get_param_count(DistributionType.Normal), NORMAL_PARAM_COUNT)

    def test_weibull(self):
        self.assertEqual(get_param_count(DistributionType.Weibull), WEIBULL_PARAM_COUNT)

    def test_gen_weibull(self):
        self.assertEqual(get_param_count(DistributionType.GeneralWeibull), GENERAL_WEIBULL_PARAM_COUNT)


# the names of parameters must be corresponding with param count
class TestGetParamNames(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(len(get_param_names(DistributionType.Normal)), NORMAL_PARAM_COUNT)

    def test_weibull(self):
        self.assertEqual(len(get_param_names(DistributionType.Weibull)), WEIBULL_PARAM_COUNT)

    def test_gen_weibull(self):
        self.assertEqual(len(get_param_names(DistributionType.GeneralWeibull)), GENERAL_WEIBULL_PARAM_COUNT)


# 1. make sure it has the func with that name
# 2. the count of params it accept is consistent with func `get_param_count`
class TestGetBaseFuncName(unittest.TestCase):
    @staticmethod
    def has_func(name):
        return name in globals().keys()

    @staticmethod
    def get_func(name):
        return globals()[name]

    def test_has_normal(self):
        func_name = get_base_func_name(DistributionType.Normal)
        self.assertTrue(self.has_func(func_name))

    def test_has_weibull(self):
        func_name = get_base_func_name(DistributionType.Weibull)
        self.assertTrue(self.has_func(func_name))

    def test_has_gen_weibull(self):
        func_name = get_base_func_name(DistributionType.GeneralWeibull)
        self.assertTrue(self.has_func(func_name))

    def test_normal_use_suitable_params(self):
        func_name = get_base_func_name(DistributionType.Normal)
        func = self.get_func(func_name)
        # the first param is x
        func(np.linspace(1, 11, 1001), *[i+1 for i in range(NORMAL_PARAM_COUNT)])

    def test_weiubll_use_suitable_params(self):
        func_name = get_base_func_name(DistributionType.Weibull)
        func = self.get_func(func_name)
        # the first param is x
        func(np.linspace(1, 11, 1001), *[i+1 for i in range(WEIBULL_PARAM_COUNT)])

    def test_gen_weibull_use_suitable_params(self):
        func_name = get_base_func_name(DistributionType.GeneralWeibull)
        func = self.get_func(func_name)
        # the first param is x
        func(np.linspace(1, 11, 1001), *[i+1 for i in range(GENERAL_WEIBULL_PARAM_COUNT)])


class TestGetParamBounds(unittest.TestCase):
    # 1. each bound must has the left and right values
    # 2. values must be real number or `None`
    # `None` means no limit
    def check_bound(self, bound):
        self.assertEqual(len(bound), 2)
        self.assertTrue(bound[0] is None == None or np.isreal(bound[0]))
        self.assertTrue(bound[1] is None == None or np.isreal(bound[1]))

    def test_normal(self):
        bounds = get_param_bounds(DistributionType.Normal)
        self.assertEqual(len(bounds), NORMAL_PARAM_COUNT)
        for bound in bounds:
            self.check_bound(bound)

    def test_weibull(self):
        bounds = get_param_bounds(DistributionType.Weibull)
        self.assertEqual(len(bounds), WEIBULL_PARAM_COUNT)
        for bound in bounds:
            self.check_bound(bound)

    def test_gen_weibull(self):
        bounds = get_param_bounds(DistributionType.GeneralWeibull)
        self.assertEqual(len(bounds), GENERAL_WEIBULL_PARAM_COUNT)
        for bound in bounds:
            self.check_bound(bound)


# length of each component's defaluts must be equal to the param count
class TestGetParamDefaults(unittest.TestCase):
    def test_normal(self):
        for component_number in range(1, 101):
            defaluts = get_param_defaults(DistributionType.Normal, component_number)
            self.assertEqual(len(defaluts), component_number)
            for defaluts_of_component in defaluts:
                self.assertEqual(len(defaluts_of_component), NORMAL_PARAM_COUNT)

    def test_weibull(self):
        for component_number in range(1, 101):
            defaluts = get_param_defaults(DistributionType.Weibull, component_number)
            self.assertEqual(len(defaluts), component_number)
            for defaluts_of_component in defaluts:
                self.assertEqual(len(defaluts_of_component), WEIBULL_PARAM_COUNT)

    def test_gen_weibull(self):
        for component_number in range(1, 101):
            defaluts = get_param_defaults(DistributionType.GeneralWeibull, component_number)
            self.assertEqual(len(defaluts), component_number)
            for defaluts_of_component in defaluts:
                self.assertEqual(len(defaluts_of_component), GENERAL_WEIBULL_PARAM_COUNT)

# 1. if COMPONENT_NUMBER equals 1, length must be PARAM_COUNT,
# else, length must be eqaul to (PARAM_COUNT+1) * COMPONENT_COUNT - 1
# (the additional param is the fraction of each component)
# 2. params have already been sorted by `location` key
class TestGetParams(unittest.TestCase):
    def check_sorted(self, params):
        for location, param in enumerate(params):
            self.assertEqual(param[LOCATION_KEY], location)

    def test_normal(self):
        for component_number in range(1, 101):
            params = get_params(DistributionType.Normal, component_number)
            if component_number == 1:
                self.assertEqual(len(params), NORMAL_PARAM_COUNT)
            else:
                self.assertEqual(len(params), (NORMAL_PARAM_COUNT+1) * component_number - 1)
            self.check_sorted(params)

    def test_weibull(self):
        for component_number in range(1, 101):
            params = get_params(DistributionType.Weibull, component_number)
            if component_number == 1:
                self.assertEqual(len(params), WEIBULL_PARAM_COUNT)
            else:
                self.assertEqual(len(params), (WEIBULL_PARAM_COUNT+1) * component_number - 1)
            self.check_sorted(params)

    def test_gen_weibull(self):
        for component_number in range(1, 101):
            params = get_params(DistributionType.GeneralWeibull, component_number)
            if component_number == 1:
                self.assertEqual(len(params), GENERAL_WEIBULL_PARAM_COUNT)
            else:
                self.assertEqual(len(params), (GENERAL_WEIBULL_PARAM_COUNT+1) * component_number - 1)
            self.check_sorted(params)


# these funcs are hard to test alone and will be called in other funcs, just call them
class TestMISC(unittest.TestCase):
    def setUp(self):
        self.normal_params = get_params(DistributionType.Normal, 10)
        self.weibull_params = get_params(DistributionType.Weibull, 10)
        self.gen_weibull_params = get_params(DistributionType.GeneralWeibull, 10)

    def tearDown(self):
        self.normal_params = None
        self.weibull_params = None
        self.gen_weibull_params = None

    def test_sort(self):
        sort_params_by_location_in_place(self.normal_params)
        sort_params_by_location_in_place(self.weibull_params)
        sort_params_by_location_in_place(self.gen_weibull_params)

    def test_get_bounds(self):
        get_bounds(self.normal_params)
        get_bounds(self.weibull_params)
        get_bounds(self.gen_weibull_params)

    def test_get_constrains(self):
        for i in range(1, 101):
            get_constrains(i)

    def test_get_defaults(self):
        get_defaults(self.normal_params)
        get_defaults(self.weibull_params)
        get_defaults(self.gen_weibull_params)


# use `exec` to check if it has syntax or other errors
class TestGetLambdaStr(unittest.TestCase):
    def test_normal(self):
        for i in range(1, 101):
            lambda_str = get_lambda_str(DistributionType.Normal, i)
            exec(lambda_str)

    def test_weibull(self):
        for i in range(1, 101):
            lambda_str = get_lambda_str(DistributionType.Weibull, i)
            exec(lambda_str)

    def test_gen_weibull(self):
        for i in range(1, 101):
            lambda_str = get_lambda_str(DistributionType.GeneralWeibull, i)
            exec(lambda_str)


# the processed params must in the form that:
# 1. component number length tuple
# 2. each tuple is consistant with one sub tuple and the fraction
# 3. the sub tuple is the params of single func that except x,
# so its length is equal to PARAM_COUNT
class TestProcessParams(unittest.TestCase):
    def test_normal(self):
        for i in range(1, 101):
            if i == 1:
                count = NORMAL_PARAM_COUNT
            else:
                count = (NORMAL_PARAM_COUNT+1)*i - 1
            fake_params = np.ones((count,))
            processed = process_params(DistributionType.Normal, i, fake_params)
            self.assertEqual(len(processed), i)
            for params, fraction in processed:
                self.assertEqual(len(params), NORMAL_PARAM_COUNT)

    def test_weibull(self):
        for i in range(1, 101):
            if i == 1:
                count = WEIBULL_PARAM_COUNT
            else:
                count = (WEIBULL_PARAM_COUNT+1)*i - 1
            fake_params = np.ones((count,))
            processed = process_params(DistributionType.Weibull, i, fake_params)
            self.assertEqual(len(processed), i)
            for params, fraction in processed:
                self.assertEqual(len(params), WEIBULL_PARAM_COUNT)

    def test_gen_weibull(self):
        for i in range(1, 101):
            if i == 1:
                count = GENERAL_WEIBULL_PARAM_COUNT
            else:
                count = (GENERAL_WEIBULL_PARAM_COUNT+1)*i - 1
            fake_params = np.ones((count,))
            processed = process_params(DistributionType.GeneralWeibull, i, fake_params)
            self.assertEqual(len(processed), i)
            for params, fraction in processed:
                self.assertEqual(len(params), GENERAL_WEIBULL_PARAM_COUNT)


# 1. PDF func return 0.0 while the param is invalid
# 2. other funcs return NaN while the param is invalid
# 3. the result values of the func generated by lambda and manuscript must be equal
class TestNormalMathFuncs(unittest.TestCase):
    @staticmethod
    def get_func(lambda_str):
        exec("func = "+lambda_str)
        return locals()["func"]

    # get zero while param is invalid
    def test_sigma_invalid(self):
        x = np.linspace(-10, 10, 1001)
        res = np.equal(normal(x, 0, -1), np.zeros_like(x))
        self.assertTrue(np.all(res))

    def test_mean_invalid(self):
        self.assertTrue(np.isnan(normal_mean(0, -1)))

    def test_median_invalid(self):
        self.assertTrue(np.isnan(normal_median(0, -1)))

    def test_mode_invalid(self):
        self.assertTrue(np.isnan(normal_mode(0, -1)))

    def test_standard_deviation_invalid(self):
        self.assertTrue(np.isnan(normal_standard_deviation(0, -1)))

    def test_variance_invalid(self):
        self.assertTrue(np.isnan(normal_variance(0, -1)))

    def test_skewness_invalid(self):
        self.assertTrue(np.isnan(normal_skewness(0, -1)))

    def test_kurtosis_invalid(self):
        self.assertTrue(np.isnan(normal_kurtosis(0, -1)))

    def test_single(self):
        lambda_str = get_lambda_str(DistributionType.Normal, 1)
        generated_func = self.get_func(lambda_str)
        manuscript_func = normal
        x = np.linspace(-10, 10, 1001)
        res = np.equal(generated_func(x, 0.7, 2.1), manuscript_func(x, 0.7, 2.1))
        self.assertTrue(np.all(res))

    def test_double(self):
        lambda_str = get_lambda_str(DistributionType.Normal, 2)
        generated_func = self.get_func(lambda_str)
        manuscript_func = double_normal
        x = np.linspace(-10, 10, 1001)
        res = np.equal(generated_func(x, 0.71, 2.41, 5.3, 12.1, 0.34), \
            manuscript_func(x, 0.71, 2.41, 5.3, 12.1, 0.34))
        self.assertTrue(np.all(res))

    def test_triple(self):
        lambda_str = get_lambda_str(DistributionType.Normal, 3)
        generated_func = self.get_func(lambda_str)
        manuscript_func = triple_normal
        x = np.linspace(-10, 10, 1001)
        res = np.equal(generated_func(x, 0.52, 1.42, 2.3, 11.2, 4.2, 12.4, 0.21, 0.42), \
            manuscript_func(x, 0.52, 1.42, 2.3, 11.2, 4.2, 12.4, 0.21, 0.42))
        self.assertTrue(np.all(res))

    def test_quadruple(self):
        lambda_str = get_lambda_str(DistributionType.Normal, 4)
        generated_func = self.get_func(lambda_str)
        manuscript_func = quadruple_normal
        x = np.linspace(-10, 10, 1001)
        res = np.equal(generated_func(x, 0.52, 1.42, 2.3, 11.2, 4.2, 12.4, 0.21, 0.46, 0.21, 0.42, 0.08), \
            manuscript_func(x, 0.52, 1.42, 2.3, 11.2, 4.2, 12.4, 0.21, 0.46, 0.21, 0.42, 0.08))
        self.assertTrue(np.all(res))


class TestWeibullMathFuncs(unittest.TestCase):
    @staticmethod
    def get_func(lambda_str):
        exec("func = "+lambda_str)
        return locals()["func"]

    def test_beta_invalid(self):
        x = np.linspace(1, 11, 1001)
        self.assertTrue(np.all(np.equal(weibull(x, 0, 1), np.zeros_like(x))))
        self.assertTrue(np.all(np.equal(weibull(x, 0.0, 1), np.zeros_like(x))))
        self.assertTrue(np.all(np.equal(weibull(x, -1, 1), np.zeros_like(x))))
        self.assertTrue(np.all(np.equal(weibull(x, -2.7, 1), np.zeros_like(x))))

    def test_eta_invalid(self):
        x = np.linspace(1, 11, 1001)
        self.assertTrue(np.all(np.equal(weibull(x, 2, 0), np.zeros_like(x))))
        self.assertTrue(np.all(np.equal(weibull(x, 2, 0.0), np.zeros_like(x))))
        self.assertTrue(np.all(np.equal(weibull(x, 2, -2), np.zeros_like(x))))
        self.assertTrue(np.all(np.equal(weibull(x, 2, -3.1), np.zeros_like(x))))

    def test_x_invalid(self):
        x = np.linspace(-2, 2, 401)
        y = weibull(x, 2, 2)
        res = np.equal(np.less_equal(x, 0.0), np.equal(y, 0.0))
        self.assertTrue(np.all(res))

    def test_mean_invalid(self):
        self.assertTrue(np.isnan(weibull_mean(-1, 1)))
        self.assertTrue(np.isnan(weibull_mean(1, -1)))

    def test_median_invalid(self):
        self.assertTrue(np.isnan(weibull_median(-1, 1)))
        self.assertTrue(np.isnan(weibull_median(1, -1)))

    def test_mode_invalid(self):
        self.assertTrue(np.isnan(weibull_mode(-1, 1)))
        self.assertTrue(np.isnan(weibull_mode(1, -1)))

    def test_standard_deviation_invalid(self):
        self.assertTrue(np.isnan(weibull_standard_deviation(-1, 1)))
        self.assertTrue(np.isnan(weibull_standard_deviation(1, -1)))

    def test_variance_invalid(self):
        self.assertTrue(np.isnan(weibull_variance(-1, 1)))
        self.assertTrue(np.isnan(weibull_variance(1, -1)))

    def test_skewness_invalid(self):
        self.assertTrue(np.isnan(weibull_skewness(-1, 1)))
        self.assertTrue(np.isnan(weibull_skewness(1, -1)))

    def test_kurtosis_invalid(self):
        self.assertTrue(np.isnan(weibull_kurtosis(-1, 1)))
        self.assertTrue(np.isnan(weibull_kurtosis(1, -1)))

    def test_single(self):
        lambda_str = get_lambda_str(DistributionType.Weibull, 1)
        generated_func = self.get_func(lambda_str)
        manuscript_func = weibull
        x = np.linspace(1, 10, 1001)
        res = np.equal(generated_func(x, 0.7, 2.1), manuscript_func(x, 0.7, 2.1))
        self.assertTrue(np.all(res))

    def test_double(self):
        lambda_str = get_lambda_str(DistributionType.Weibull, 2)
        generated_func = self.get_func(lambda_str)
        manuscript_func = double_weibull
        x = np.linspace(1, 10, 1001)
        res = np.equal(generated_func(x, 0.71, 2.41, 5.3, 12.1, 0.34), \
            manuscript_func(x, 0.71, 2.41, 5.3, 12.1, 0.34))
        self.assertTrue(np.all(res))

    def test_triple(self):
        lambda_str = get_lambda_str(DistributionType.Weibull, 3)
        generated_func = self.get_func(lambda_str)
        manuscript_func = triple_weibull
        x = np.linspace(1, 10, 1001)
        res = np.equal(generated_func(x, 0.52, 1.42, 2.3, 11.2, 4.2, 12.4, 0.21, 0.42), \
            manuscript_func(x, 0.52, 1.42, 2.3, 11.2, 4.2, 12.4, 0.21, 0.42))
        self.assertTrue(np.all(res))

    def test_quadruple(self):
        lambda_str = get_lambda_str(DistributionType.Weibull, 4)
        generated_func = self.get_func(lambda_str)
        manuscript_func = quadruple_weibull
        x = np.linspace(1, 10, 1001)
        res = np.equal(generated_func(x, 0.52, 1.42, 2.3, 11.2, 4.2, 12.4, 0.21, 0.46, 0.21, 0.42, 0.08), \
            manuscript_func(x, 0.52, 1.42, 2.3, 11.2, 4.2, 12.4, 0.21, 0.46, 0.21, 0.42, 0.08))
        self.assertTrue(np.all(res))


class TestGeneralWeibullMathFuncs(unittest.TestCase):
    @staticmethod
    def get_func(lambda_str):
        exec("func = "+lambda_str)
        return locals()["func"]

    def test_base(self):
        x1 = np.linspace(1, 11, 1001)
        x_offset = 2.458
        x2 = x1 - x_offset
        res = np.equal(weibull(x2, 2.786, 5.267), gen_weibull(x1, x_offset, 2.786, 5.267))
        self.assertTrue(np.all(res))

    def test_mean(self):
        self.assertEqual(weibull_mean(2.144, 2.455), gen_weibull_mean(0, 2.144, 2.455))

    def test_median(self):
        self.assertEqual(weibull_median(2.144, 2.455), gen_weibull_median(0, 2.144, 2.455))

    def test_mode(self):
        self.assertEqual(weibull_mode(2.144, 2.455), gen_weibull_mode(0, 2.144, 2.455))

    def test_standard_deviation(self):
        self.assertEqual(weibull_standard_deviation(2.144, 2.455), gen_weibull_standard_deviation(0, 2.144, 2.455))

    def test_variance(self):
        self.assertEqual(weibull_variance(2.144, 2.455), gen_weibull_variance(0, 2.144, 2.455))

    def test_skewness(self):
        self.assertEqual(weibull_skewness(2.144, 2.455), gen_weibull_skewness(0, 2.144, 2.455))

    def test_kurtosis(self):
        self.assertEqual(weibull_kurtosis(2.144, 2.455), gen_weibull_kurtosis(0, 2.144, 2.455))

    def test_single(self):
        lambda_str = get_lambda_str(DistributionType.GeneralWeibull, 1)
        generated_func = self.get_func(lambda_str)
        manuscript_func = gen_weibull
        x = np.linspace(1, 10, 1001)
        res = np.equal(generated_func(x, 0.2, 0.7, 2.1), manuscript_func(x, 0.2, 0.7, 2.1))
        self.assertTrue(np.all(res))

    def test_double(self):
        lambda_str = get_lambda_str(DistributionType.GeneralWeibull, 2)
        generated_func = self.get_func(lambda_str)
        manuscript_func = double_gen_weibull
        x = np.linspace(1, 10, 1001)
        res = np.equal(generated_func(x, 0.37, 0.71, 2.41, 0.45, 5.3, 12.1, 0.34), \
            manuscript_func(x, 0.37, 0.71, 2.41, 0.45, 5.3, 12.1, 0.34))
        self.assertTrue(np.all(res))

    def test_triple(self):
        lambda_str = get_lambda_str(DistributionType.GeneralWeibull, 3)
        generated_func = self.get_func(lambda_str)
        manuscript_func = triple_gen_weibull
        x = np.linspace(1, 10, 1001)
        res = np.equal(generated_func(x, 0.76, 0.52, 1.42, 0.65, 2.3, 11.2, 0.54, 4.2, 12.4, 0.21, 0.42), \
            manuscript_func(x, 0.76, 0.52, 1.42, 0.65, 2.3, 11.2, 0.54, 4.2, 12.4, 0.21, 0.42))
        self.assertTrue(np.all(res))

    def test_quadruple(self):
        lambda_str = get_lambda_str(DistributionType.GeneralWeibull, 4)
        generated_func = self.get_func(lambda_str)
        manuscript_func = quadruple_gen_weibull
        x = np.linspace(1, 10, 1001)
        res = np.equal(generated_func(x, 0.80, 0.52, 1.42, 0.21, 2.3, 11.2, 0.43, 4.2, 12.4, 0.76, 0.21, 0.46, 0.21, 0.42, 0.08), \
            manuscript_func(x, 0.80, 0.52, 1.42, 0.21, 2.3, 11.2, 0.43, 4.2, 12.4, 0.76, 0.21, 0.46, 0.21, 0.42, 0.08))
        self.assertTrue(np.all(res))


# the func must be corresponding to the name
class TestGetSingleFunc(unittest.TestCase):
    @staticmethod
    def get_func(distribution_type: DistributionType):
        name = get_base_func_name(distribution_type)
        return globals()[name]

    def test_normal(self):
        actual_func = get_single_func(DistributionType.Normal)
        expected_func = self.get_func(DistributionType.Normal)
        self.assertIs(actual_func, expected_func)

    def test_weibull(self):
        actual_func = get_single_func(DistributionType.Weibull)
        expected_func = self.get_func(DistributionType.Weibull)
        self.assertIs(actual_func, expected_func)

    def test_gen_weibull(self):
        actual_func = get_single_func(DistributionType.GeneralWeibull)
        expected_func = self.get_func(DistributionType.GeneralWeibull)
        self.assertIs(actual_func, expected_func)


# the length of return values must be same as that of `get_defaults`
class TestGetParamByMean(unittest.TestCase):
    def test_normal(self):
        for i in range(1, 101):
            params = get_param_by_mean(DistributionType.Normal, i, np.linspace(1, 10, i))
            if i == 0:
                self.assertEqual(len(params), NORMAL_PARAM_COUNT)
            else:
                self.assertEqual(len(params), (NORMAL_PARAM_COUNT+1)*i - 1)

    def test_weibull(self):
        for i in range(1, 101):
            params = get_param_by_mean(DistributionType.Weibull, i, np.linspace(1, 10, i))
            if i == 0:
                self.assertEqual(len(params), WEIBULL_PARAM_COUNT)
            else:
                self.assertEqual(len(params), (WEIBULL_PARAM_COUNT+1)*i - 1)

    def test_gen_weibull(self):
        for i in range(1, 101):
            params = get_param_by_mean(DistributionType.GeneralWeibull, i, np.linspace(1, 10, i))
            if i == 0:
                self.assertEqual(len(params), GENERAL_WEIBULL_PARAM_COUNT)
            else:
                self.assertEqual(len(params), (GENERAL_WEIBULL_PARAM_COUNT+1)*i - 1)


class TestAlgorithmData(unittest.TestCase):
    def test_ctor(self):
        for i in range(1, 101):
            n = AlgorithmData(DistributionType.Normal, i)
            w = AlgorithmData(DistributionType.Weibull, i)
            g = AlgorithmData(DistributionType.GeneralWeibull, i)

    def setUp(self):
        self.normal_data = AlgorithmData(DistributionType.Normal, 10)
        self.weibull_data = AlgorithmData(DistributionType.Weibull, 10)
        self.gen_weibull_data = AlgorithmData(DistributionType.GeneralWeibull, 10)

    def tearDown(self):
        self.normal_data = None
        self.weibull_data = None
        self.gen_weibull_data = None

    # these attrs will be used in other files
    def test_has_attrs(self):
        for data in [self.normal_data, self.weibull_data, self.gen_weibull_data]:
            data.distribution_type
            data.component_number
            data.param_count
            data.param_names
            data.single_func
            data.mixed_func
            data.bounds
            data.defaults
            data.constrains
            data.mean
            data.median
            data.mode
            data.variance
            data.standard_deviation
            data.skewness
            data.kurtosis

    def test_read_only(self):
        for data in [self.normal_data, self.weibull_data, self.gen_weibull_data]:
            with self.assertRaises(AttributeError):
                data.distribution_type = None
            with self.assertRaises(AttributeError):
                data.component_number = None
            with self.assertRaises(AttributeError):
                data.param_count = None
            with self.assertRaises(AttributeError):
                data.param_names = None
            with self.assertRaises(AttributeError):
                data.single_func = None
            with self.assertRaises(AttributeError):
                data.mixed_func = None
            with self.assertRaises(AttributeError):
                data.bounds = None
            with self.assertRaises(AttributeError):
                data.defaults = None
            with self.assertRaises(AttributeError):
                data.constrains = None
            with self.assertRaises(AttributeError):
                data.mean = None
            with self.assertRaises(AttributeError):
                data.median = None
            with self.assertRaises(AttributeError):
                data.mode = None
            with self.assertRaises(AttributeError):
                data.variance = None
            with self.assertRaises(AttributeError):
                data.standard_deviation = None
            with self.assertRaises(AttributeError):
                data.skewness = None
            with self.assertRaises(AttributeError):
                data.kurtosis = None

    def test_process_params(self):
        for data in [self.normal_data, self.weibull_data, self.gen_weibull_data]:
            func_params = get_params(data.distribution_type, data.component_number)
            fake_params = get_defaults(func_params)
            actual_1 = data.process_params(fake_params, 0.0)
            actual_2 = data.process_params(fake_params, 3.1)
            expected = process_params(data.distribution_type, data.component_number, fake_params)
            self.assertEqual(actual_1, expected)
            if data.distribution_type == DistributionType.Normal or \
                data.distribution_type == DistributionType.GeneralWeibull:
                self.assertNotEqual(actual_2, expected)

    def test_get_param_by_mean(self):
        for data in [self.normal_data, self.weibull_data, self.gen_weibull_data]:
            data.get_param_by_mean(np.linspace(1, 10, data.component_number))

if __name__ == "__main__":
    unittest.main()
