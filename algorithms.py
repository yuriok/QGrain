from enum import Enum, unique
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import gamma

INFINITESIMAL = 1e-100

@unique
class DistributionType(Enum):
    Normal = 0
    Weibull = 1


def check_component_number(component_number: int):
    # Check the validity of `component_number`
    if type(component_number) != int:
        raise TypeError(component_number)
    if component_number < 1:
        raise ValueError(component_number)

def get_param_names(distribution_type: DistributionType) -> Tuple[str]:
    if distribution_type == DistributionType.Normal:
        return ("mu", "sigma")
    elif distribution_type == DistributionType.Weibull:
        return ("beta", "eta")
    else:
        raise NotImplementedError(distribution_type)

def get_params(component_number: int, distribution_type: DistributionType) -> List[Dict]:
    params = []
    param1_name, param2_name = get_param_names(distribution_type)
    if component_number == 1:
        # if there is only one component, the fraction is not needful
        # beta and eta also don't need the number to distinguish
        if distribution_type == DistributionType.Normal:
            params.append({"name": param1_name, "default": 10, "location": 0, "bounds": (None, None)})
            params.append({"name": param2_name, "default": 3, "location": 1, "bounds": (INFINITESIMAL, None)})
        elif distribution_type == DistributionType.Weibull:
            params.append({"name": param1_name, "default": 3, "location": 0, "bounds": (INFINITESIMAL+2, None)})
            params.append({"name": param2_name, "default": 10, "location": 1, "bounds": (INFINITESIMAL, None)})
        else:
            raise NotImplementedError(distribution_type)
    elif component_number > 1:
        for i in range(component_number):
            # the shape params of each distribution
            # it performs better while the params between components are different
            if distribution_type == DistributionType.Normal:
                params.append({"name": "{0}{1}".format(param1_name, i+1), "default": (i+1)*10, "location": i*2, "bounds": (None, None)})
                params.append({"name": "{0}{1}".format(param2_name, i+1), "default": 2+i, "location": i*2 + 1, "bounds": (INFINITESIMAL, None)})
            elif distribution_type == DistributionType.Weibull:
                params.append({"name": "{0}{1}".format(param1_name, i+1), "default": 2+i, "location": i*2, "bounds": (INFINITESIMAL+2, None)})
                params.append({"name": "{0}{1}".format(param2_name, i+1), "default": (i+1)*10, "location": i*2 + 1, "bounds": (INFINITESIMAL, None)})
            else:
                raise NotImplementedError(distribution_type)
        for i in range(component_number-1):
            # the fraction of each distribution
            params.append({"name": "f{0}".format(i+1), "default": 1/component_number, "location": component_number*2 + i, "bounds": (0, 1)})
    else:
        raise ValueError(component_number)
    return params

def sort_params_by_location_in_place(params: List[Dict]) -> List[Dict]:
    params.sort(key=lambda element: element["location"])

def get_bounds(params: List[Dict]) -> List[Tuple]:
    bounds = []
    for param in params:
        bounds.append(param["bounds"])
    return bounds

def get_constrains(component_number: int) -> List[Dict]:
    if component_number == 1:
        return []
    elif component_number > 1:
        return [{'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-component_number:]) + INFINITESIMAL}]
    else:
        raise ValueError(component_number)

def get_defaults(params: List[Dict]) -> List:
    defaults = []
    for param in params:
        defaults.append(param["default"])
    return defaults

def get_lambda_string(component_number:int, params: List[Dict], distribution_type: DistributionType) -> str:
    param1_name, param2_name = get_param_names(distribution_type)
    if distribution_type == DistributionType.Normal:
        base_func_name = "normal"
    elif distribution_type == DistributionType.Weibull:
        base_func_name = "weibull"
    else:
        raise NotImplementedError(distribution_type)
    if component_number == 1:
        return "lambda x, {0}, {1}: {2}(x, {0}, {1})".format(param1_name, param2_name, base_func_name)
    elif component_number > 1:
        parameter_list = ", ".join(["x"] + [param["name"] for param in params])
        # " + " to connect each sub-function
        # the previous sub-function str list means the n-1 sub-functions `fi * func(x, ai, bi)`
        # the last sub-function str which represents `(1-fi-...-fn-1)*func(x, an, bn)`
        previous_sub_func_strs = ["f{0}*{1}(x, {2}{0}, {3}{0})".format(i+1, base_func_name, param1_name, param2_name) for i in range(component_number-1)]
        last_sub_func_str = "({0})*{1}(x, {2}{3}, {4}{3})".format("-".join(["1"]+["f{0}".format(i+1) for i in range(component_number-1)]), base_func_name, param1_name, component_number, param2_name)
        expression = " + ".join(previous_sub_func_strs + [last_sub_func_str])
        lambda_string = "lambda {0}: {1}".format(parameter_list, expression)
        return lambda_string
    else:
        raise ValueError(component_number)

# prcess the raw params list to make it easy to use
def process_params(component_number: int, func_params: List[Dict], fitted_params: List, distribution_type: DistributionType) -> List[List]:
    param1_name, param2_name = get_param_names(distribution_type)
    if component_number == 1:
        assert len(fitted_params) == 2
        the_only_component = [None, None, 1]
        for func_param in func_params:
            if func_param["name"] == param1_name:
                the_only_component[0] = fitted_params[func_param["location"]]
            elif func_param["name"] == param2_name:
                the_only_component[1] = fitted_params[func_param["location"]]
            else:
                raise ValueError(func_param)
        return [the_only_component]
    elif component_number > 1:
        # initialize the result list
        processed = []
        for i in range(component_number):
            processed.append([None, None, 1])
        
        for func_param in func_params:
            name = func_param["name"] # type: str
            if name.startswith(param1_name):
                comp_index = int(name[len(param1_name):]) - 1
                processed[comp_index][0] = fitted_params[func_param["location"]]
            elif name.startswith(param2_name):
                comp_index = int(name[len(param2_name):]) - 1
                processed[comp_index][1] = fitted_params[func_param["location"]]
            elif name.startswith("f"):
                comp_index = int(name[1:]) - 1
                processed[comp_index][2] = fitted_params[func_param["location"]]
                processed[-1][2] -= fitted_params[func_param["location"]]
            else:
                raise ValueError(func_param)
        # sort the list by mean
        if distribution_type == DistributionType.Normal:
            processed.sort(key=lambda element: normal_mean(*element[:-1]))
        elif distribution_type == DistributionType.Weibull:
            processed.sort(key=lambda element: weibull_mean(*element[:-1]))
        else:
            raise NotImplementedError(distribution_type)
        return processed
    else:
        raise ValueError(component_number)


# the pdf function of Normal distribution
def normal(x, mu, sigma):
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
    return 3


# The pdf function of Weibull distribution
def weibull(x, beta, eta):
    return beta/eta*(x / eta)**(beta-1) * np.exp(-(x/eta)**beta)

def double_weibull(x, beta1, eta1, beta2, eta2, f):
    return f*weibull(x, beta1, eta1) + (1-f)*weibull(x, beta2, eta2)

def triple_weibull(x, beta1, eta1, beta2, eta2, beta3, eta3, f1, f2):
    return f1*weibull(x, beta1, eta1) + f2*weibull(x, beta2, eta2) + (1-f1-f2)*weibull(x, beta3, eta3)

def quadruple_weibull(x, beta1, eta1, beta2, eta2, beta3, eta3, beta4, eta4, f1, f2, f3):
    return f1*weibull(x, beta1, eta1) + f2*weibull(x, beta2, eta2) + f3*weibull(x, beta3, eta3) + (1-f1-f2-f3)*weibull(x, beta4, eta4)

def weibull_mean(beta, eta):
    return eta*gamma(1/beta+1)

def weibull_median(beta, eta):
    return eta*(np.log(2)**(1/beta))

def weibull_mode(beta, eta):
    return eta*(1-1/beta)**(1/beta)

def weibull_standard_deviation(beta, eta):
    return eta*np.sqrt(gamma(2/beta+1) - gamma(1/beta+1)**2)

def weibull_variance(beta, eta):
    return (eta**2)*(gamma(2/beta+1)-gamma(1/beta+1)**2)

def weibull_skewness(beta, eta):
    return (2*gamma(1/beta+1)**3 - 3*gamma(2/beta+1)*gamma(1/beta+1) + gamma(3/beta+1)) / (gamma(2/beta+1)-gamma(1/beta+1)**2)**(3/2)

def weibull_kurtosis(beta, eta):
    return (-3*gamma(1/beta+1)**4 + 6*gamma(2/beta+1)*gamma(1/beta+1)**2 - 4*gamma(3/beta+1)*gamma(1/beta+1) + gamma(4/beta+1)) / (gamma(2/beta+1)-gamma(1/beta+1)**2)**2


def get_single_func(distribution_type: DistributionType) -> callable:
    if distribution_type == DistributionType.Normal:
        return normal
    elif distribution_type == DistributionType.Weibull:
        return weibull
    else:
        raise NotImplementedError(distribution_type)

class MixedDistributionData:
    def __init__(self, component_number: int, distribution_type: DistributionType):
        check_component_number(component_number)
        local_params = {"__tempmMixedFunc": None}
        func_params = get_params(component_number, distribution_type)
        sort_params_by_location_in_place(func_params)
        lambda_string = get_lambda_string(component_number, func_params, distribution_type)
        bounds = get_bounds(func_params)
        constrains = get_constrains(component_number)
        defaults = get_defaults(func_params)
        exec("__tempmMixedFunc=" + lambda_string, None, local_params)

        self.component_number = component_number
        self.distribution_type = distribution_type
        self.mixed_func = local_params["__tempmMixedFunc"]
        self.single_func = get_single_func(distribution_type)
        self.bounds = bounds
        self.constrains = constrains
        self.defaults = defaults
        self.func_params = func_params
        self.get_statistic_func()

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
        else:
            raise NotImplementedError(distribution_type)
    
    def process_params(self, fitted_params: List) -> List[List]:
        return process_params(self.component_number, self.func_params, fitted_params, self.distribution_type)
