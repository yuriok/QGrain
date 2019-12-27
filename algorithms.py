from enum import Enum, unique
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from scipy.special import gamma

INFINITESIMAL = 1e-100
FRACTION_PARAM_NAME = "f"
NAME_KEY = "Name"
BOUNDS_KEY = "Bounds"
DEFAULT_VALUE_KEY = "Default"
LOCATION_KEY = "Location"
COMPONENT_INDEX_KEY = "ComponentIndex"
PARAM_INDEX_KEY = "ParamIndex"


@unique
class DistributionType(Enum):
    Normal = 0
    Weibull = 1
    GeneralWeibull = 2


def check_component_number(component_number: int):
    # Check the validity of `component_number`
    if type(component_number) != int:
        raise TypeError(component_number)
    elif component_number < 1:
        raise ValueError(component_number)

def get_param_count(distribution_type: DistributionType) -> int:
    if distribution_type == DistributionType.Normal:
        return 2
    elif distribution_type == DistributionType.Weibull:
        return 2
    elif distribution_type == DistributionType.GeneralWeibull:
        return 3
    else:
        raise NotImplementedError(distribution_type)

def get_param_names(distribution_type: DistributionType) -> Tuple[str]:
    if distribution_type == DistributionType.Normal:
        return ("mu", "sigma")
    elif distribution_type == DistributionType.Weibull:
        return ("beta", "eta")
    elif distribution_type == DistributionType.GeneralWeibull:
        return ("mu", "beta", "eta")
    else:
        raise NotImplementedError(distribution_type)

def get_base_func_name(distribution_type: DistributionType) -> str:
    if distribution_type == DistributionType.Normal:
        return "normal"
    elif distribution_type == DistributionType.Weibull:
        return "weibull"
    elif distribution_type == DistributionType.GeneralWeibull:
        return "gen_weibull"
    else:
        raise NotImplementedError(distribution_type)

def get_param_bounds(distribution_type: DistributionType) -> Tuple[Tuple[float, float]]:
    if distribution_type == DistributionType.Normal:
        return ((None, None), (INFINITESIMAL, None))
    elif distribution_type == DistributionType.Weibull:
        return ((INFINITESIMAL, None), (INFINITESIMAL, None))
    elif distribution_type == DistributionType.GeneralWeibull:
        return ((None, None), (INFINITESIMAL, None), (INFINITESIMAL, None))
    else:
        raise NotImplementedError(distribution_type)

# in order to obtain better performance,
# the params of components should be different
def get_param_defaults(distribution_type: DistributionType, component_number: int) -> Tuple[Tuple]:
    check_component_number(component_number)
    defaults = []
    if distribution_type == DistributionType.Normal:
        return tuple(((i*10, 2+i) for i in range(1, component_number+1)))
    elif distribution_type == DistributionType.Weibull:
        return tuple(((10+i, (i+1)*15) for i in range(1, component_number+1)))
    elif distribution_type == DistributionType.GeneralWeibull:
        return tuple(((0, 2+i, i*10) for i in range(1, component_number+1)))
    else:
        raise NotImplementedError(distribution_type)

def get_params(distribution_type: DistributionType, component_number: int) -> List[Dict]:
    check_component_number(component_number)
    params = []
    param_count = get_param_count(distribution_type)
    param_names = get_param_names(distribution_type)
    param_bounds = get_param_bounds(distribution_type)
    param_defaults = get_param_defaults(distribution_type, component_number)
    # generate params for all components
    for component_index, component_defaults in enumerate(param_defaults):
        for param_index, name, bounds, defalut in zip(range(param_count), param_names, param_bounds, component_defaults):
            params.append({NAME_KEY: name+str(component_index+1), BOUNDS_KEY: bounds,
                           DEFAULT_VALUE_KEY: defalut, COMPONENT_INDEX_KEY: component_index,
                           PARAM_INDEX_KEY: param_index, LOCATION_KEY: component_index*param_count+param_index})
    # generate fractions for front n-1 components
    for component_index in range(component_number-1):
        # the fraction of each distribution
        params.append({NAME_KEY: FRACTION_PARAM_NAME+str(component_index+1), BOUNDS_KEY: (0, 1),
                       DEFAULT_VALUE_KEY: 1/component_number, COMPONENT_INDEX_KEY: component_index,
                       LOCATION_KEY: component_number*param_count + component_index})
    sort_params_by_location_in_place(params)
    return params

def sort_params_by_location_in_place(params: List[Dict]):
    params.sort(key=lambda element: element[LOCATION_KEY])

def get_bounds(params: List[Dict]) -> List[Tuple]:
    bounds = []
    for param in params:
        bounds.append(param[BOUNDS_KEY])
    return bounds

def get_constrains(component_number: int) -> Tuple[Dict]:
    if component_number == 1:
        return ()
    elif component_number > 1:
        return ({'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-component_number:]) + INFINITESIMAL})
    else:
        raise ValueError(component_number)

def get_defaults(params: List[Dict]) -> List:
    defaults = []
    for param in params:
        defaults.append(param[DEFAULT_VALUE_KEY])
    return defaults

def get_lambda_str(distribution_type: DistributionType, component_number:int) -> str:
    base_func_name = get_base_func_name(distribution_type)
    param_count = get_param_count(distribution_type)
    param_names = get_param_names(distribution_type)
    if component_number == 1:
        return "lambda x, {0}: {1}(x, {0})".format(", ".join(param_names), base_func_name)
    elif component_number > 1:
        parameter_list = ", ".join(["x"] + [name+str(i+1) for i in range(component_number) for name in param_names] + [FRACTION_PARAM_NAME+str(i+1) for i in range(component_number-1)])
        # " + " to connect each sub-function
        # the previous sub-function str list means the m-1 sub-functions with n params `fj * base_func(x, param_1_j, ..., param_i_j, ..., param_n_j)`
        # the last sub-function str which represents `(1-f_1-...-f_j-...-f_m-1) * base_func(x, param_1_j, ..., param_i_j, ..., param_n_j)`
        previous_format_str = "{0}{1}*{2}(x, " + ", ".join(["{"+str(i+3)+"}{1}" for i in range(param_count)]) + ")"
        previous_sub_func_strs = [previous_format_str.format(FRACTION_PARAM_NAME, i+1, base_func_name, *param_names) for i in range(component_number-1)]
        last_format_str = "({0})*{1}(x, " + ", ".join(["{"+str(i+3)+"}{2}" for i in range(param_count)]) + ")"
        last_sub_func_str = last_format_str.format("-".join(["1"]+["f{0}".format(i+1) for i in range(component_number-1)]), base_func_name, component_number, *param_names)
        expression = " + ".join(previous_sub_func_strs + [last_sub_func_str])
        lambda_string = "lambda {0}: {1}".format(parameter_list, expression)
        return lambda_string
    else:
        raise ValueError(component_number)

# prcess the raw params list to make it easy to use
def process_params(distribution_type: DistributionType, component_number: int, fitted_params: Iterable) -> Tuple[Tuple[Tuple, float]]:
    param_count = get_param_count(distribution_type)
    if component_number == 1:
        assert len(fitted_params) == param_count
        return ((tuple(fitted_params), 1.0))
    elif component_number > 1:
        assert len(fitted_params) == (param_count+1) * component_number - 1
        expanded = list(fitted_params) + [1.0-sum(fitted_params[component_number*param_count:])]
        return tuple(((tuple(expanded[i*param_count:(i+1)*param_count]), expanded[component_number*param_count+i]) for i in range(component_number)))
    else:
        raise ValueError(component_number)

# the pdf function of Normal distribution
def normal(x, mu, sigma):
    if sigma <= 0.0:
        return np.zeros_like(x, dtype=np.float64)
    else:
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-np.square(x-mu)/(2*np.square(sigma)))

def double_normal(x, mu1, sigma1, mu2, sigma2, f1):
    return f1 * normal(x, mu1, sigma1) + (1-f1) * normal(x, mu2, sigma2)

def triple_normal(x, mu1, sigma1, mu2, sigma2, mu3, sigma3, f1, f2):
    return f1 * normal(x, mu1, sigma1) + f2 * normal(x, mu2, sigma2) + (1-f1-f2) * normal(x, mu3, sigma3)

def quadruple_normal(x, mu1, sigma1, mu2, sigma2, mu3, sigma3, mu4, sigma4, f1, f2, f3):
    return f1 * normal(x, mu1, sigma1) + f2 * normal(x, mu2, sigma2) + f3 * normal(x, mu3, sigma3) + (1-f1-f2-f3) * normal(x, mu4, sigma4)

def normal_mean(mu, sigma):
    return mu

def normal_median(mu, sigma):
    return mu

def normal_mode(mu, sigma):
    return mu

def normal_standard_deviation(mu, sigma):
    return sigma

def normal_variance(mu, sigma):
    return sigma**2

def normal_skewness(mu, sigma):
    return 0

def normal_kurtosis(mu, sigma):
    return 0


# The pdf function of Weibull distribution
def weibull(x, beta, eta):
    results = np.zeros_like(x, dtype=np.float64)
    if beta <= 0.0 or eta <= 0.0:
        return results
    else:
        non_zero = np.greater(x, 0.0)
        results[non_zero] = (beta/eta) * (x[non_zero]/eta)**(beta-1) * np.exp(-(x[non_zero]/eta)**beta)
        return results
    # return (beta/eta) * (x/eta)**(beta-1) * np.exp(-(x/eta)**beta)

def double_weibull(x, beta1, eta1, beta2, eta2, f):
    return f * weibull(x, beta1, eta1) + (1-f) * weibull(x, beta2, eta2)

def triple_weibull(x, beta1, eta1, beta2, eta2, beta3, eta3, f1, f2):
    return f1 * weibull(x, beta1, eta1) + f2 * weibull(x, beta2, eta2) + (1-f1-f2) * weibull(x, beta3, eta3)

def quadruple_weibull(x, beta1, eta1, beta2, eta2, beta3, eta3, beta4, eta4, f1, f2, f3):
    return f1 * weibull(x, beta1, eta1) + f2 * weibull(x, beta2, eta2) + f3 * weibull(x, beta3, eta3) + (1-f1-f2-f3) * weibull(x, beta4, eta4)

def weibull_mean(beta, eta):
    return eta*gamma(1/beta+1)

def weibull_median(beta, eta):
    return eta*(np.log(2)**(1/beta))

def weibull_mode(beta, eta):
    if beta <= 1:
        return 0.0
    else:
        return eta*(1-1/beta)**(1/beta)

def weibull_standard_deviation(beta, eta):
    return eta*np.sqrt(gamma(2/beta+1) - gamma(1/beta+1)**2)

def weibull_variance(beta, eta):
    return (eta**2)*(gamma(2/beta+1)-gamma(1/beta+1)**2)

def weibull_skewness(beta, eta):
    return (2*gamma(1/beta+1)**3 - 3*gamma(2/beta+1)*gamma(1/beta+1) + gamma(3/beta+1)) / (gamma(2/beta+1)-gamma(1/beta+1)**2)**(3/2)

def weibull_kurtosis(beta, eta):
    return (-3*gamma(1/beta+1)**4 + 6*gamma(2/beta+1)*gamma(1/beta+1)**2 - 4*gamma(3/beta+1)*gamma(1/beta+1) + gamma(4/beta+1)) / (gamma(2/beta+1)-gamma(1/beta+1)**2)**2


def gen_weibull(x, mu, beta, eta):
    return weibull(x-mu, beta, eta)

def double_gen_weibull(x, mu1, beta1, eta1, mu2, beta2, eta2, f):
    return f * gen_weibull(x, mu1, beta1, eta1) + (1-f) * gen_weibull(x, mu2, beta2, eta2)

def triple_gen_weibull(x, mu1, beta1, eta1, mu2, beta2, eta2, mu3, beta3, eta3, f1, f2):
    return f1 * gen_weibull(x, mu1, beta1, eta1) + f2 * gen_weibull(x, mu2, beta2, eta2) + (1-f1-f2)*gen_weibull(x, mu3, beta3, eta3)

def quadruple_gen_weibull(x, mu1, beta1, eta1, mu2, beta2, eta2, mu3, beta3, eta3, mu4, beta4, eta4, f1, f2, f3):
    return f1 * gen_weibull(x, mu1, beta1, eta1) + f2 * gen_weibull(x, mu2, beta2, eta2) + f3 * gen_weibull(x, mu3, beta3, eta3) + (1-f1-f2-f3) * gen_weibull(x, mu4, beta4, eta4)

def gen_weibull_mean(mu, beta, eta):
    return weibull_mean(beta, eta) + mu

def gen_weibull_median(mu, beta, eta):
    return weibull_median(beta, eta) + mu

def gen_weibull_mode(mu, beta, eta):
    return weibull_mode(beta, eta) + mu

def gen_weibull_standard_deviation(mu, beta, eta):
    return weibull_standard_deviation(beta, eta)

def gen_weibull_variance(mu, beta, eta):
    return weibull_variance(beta, eta)

def gen_weibull_skewness(mu, beta, eta):
    return weibull_skewness(beta, eta)

def gen_weibull_kurtosis(mu, beta, eta):
    return weibull_kurtosis(beta, eta)


def get_single_func(distribution_type: DistributionType) -> Callable:
    if distribution_type == DistributionType.Normal:
        return normal
    elif distribution_type == DistributionType.Weibull:
        return weibull
    elif distribution_type == DistributionType.GeneralWeibull:
        return gen_weibull
    else:
        raise NotImplementedError(distribution_type)

def get_param_by_mean(distribution_type: DistributionType, component_number: int, mean_values: Iterable):
    assert len(mean_values) == component_number
    param_count = get_param_count(distribution_type)
    func_params = get_params(distribution_type, component_number)
    param_values = get_defaults(func_params)
    if distribution_type == DistributionType.Normal:
        for i in range(component_number):
            # for normal distribution
            # only change the loaction param (first param of each component)
            param_values[i*param_count] = mean_values[i]
    elif distribution_type == DistributionType.Weibull:
        for i in range(component_number):

            beta = param_values[i*param_count]
            param_values[i*param_count+1] = mean_values[i] / gamma(1/beta+1)
    elif distribution_type == DistributionType.GeneralWeibull:
        for i in range(component_number):
            mu = param_values[i*param_count]
            beta = param_values[i*param_count+1]
            param_values[i*param_count+2] = (mean_values[i]-mu) / gamma(1/beta+1)
    else:
        raise NotImplementedError(distribution_type)
    return param_values


class AlgorithmData:
    def __init__(self, distribution_type: DistributionType, component_number: int):
        check_component_number(component_number)
        self.lambda_str = get_lambda_str(distribution_type, component_number)
        self.mixed_func = self.get_func_by_lambda_str(self.lambda_str)
        self.func_params = get_params(distribution_type, component_number)
        self.bounds = get_bounds(self.func_params)
        self.defaults = get_defaults(self.func_params)
        self.constrains = get_constrains(component_number)
        self.single_func = get_single_func(distribution_type)
        self.component_number = component_number
        self.distribution_type = distribution_type
        self.get_statistic_func()

    def get_func_by_lambda_str(self, lambda_str: str) -> Callable:
        local_params = {"__tempmMixedFunc": None}
        exec("__tempmMixedFunc=" + lambda_str, None, local_params)
        mixed_func = local_params["__tempmMixedFunc"]
        return mixed_func

    def get_statistic_func(self):
        if self.distribution_type == DistributionType.Normal:
            self.mean = normal_mean
            self.median = normal_median
            self.mode = normal_mode
            self.standard_deviation = normal_standard_deviation
            self.variance = normal_variance
            self.skewness = normal_skewness
            self.kurtosis = normal_kurtosis
        elif self.distribution_type == DistributionType.Weibull:
            self.mean = weibull_mean
            self.median = weibull_median
            self.mode = weibull_mode
            self.standard_deviation = weibull_standard_deviation
            self.variance = weibull_variance
            self.skewness = weibull_skewness
            self.kurtosis = weibull_kurtosis
        elif self.distribution_type == DistributionType.GeneralWeibull:
            self.mean = gen_weibull_mean
            self.median = gen_weibull_median
            self.mode = gen_weibull_mode
            self.standard_deviation = gen_weibull_standard_deviation
            self.variance = gen_weibull_variance
            self.skewness = gen_weibull_skewness
            self.kurtosis = gen_weibull_kurtosis
        else:
            raise NotImplementedError(distribution_type)

    def get_param_names(self) -> Tuple[str]:
        return get_param_names(self.distribution_type)

    def get_param_count(self) -> int:
        return get_param_count(self.distribution_type)

    def process_params(self, fitted_params: Iterable, x_offset: float) -> Tuple[Tuple[Tuple, float]]:
        params_copy = np.array(fitted_params)
        param_count = get_param_count(self.distribution_type)
        if self.distribution_type == DistributionType.Normal or self.distribution_type == DistributionType.GeneralWeibull:
            for i in range(self.component_number):
                params_copy[i*param_count] += x_offset
        return process_params(self.distribution_type, self.component_number, params_copy)

    def get_param_by_mean(self, mean_values: Iterable):
        return get_param_by_mean(self.distribution_type, self.component_number, mean_values)
