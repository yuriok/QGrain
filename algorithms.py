import numpy as np
from scipy.special import gamma
from enum import Enum, unique

INFINITESIMAL = 1e-100

@unique
class DistributionType(Enum):
    LogNormal = 1
    Weibull = 2


def check_ncomp(ncomp: int):
    # Check the validity of `ncomp`
    if type(ncomp) != int:
        raise TypeError(ncomp)
    if ncomp <= 0:
        raise ValueError(ncomp)


def get_params(ncomp: int, distribution_type: DistributionType) -> list:
    params = []
    if ncomp == 1:
        # if there is only one component, the fraction is not needful
        # beta and eta also don't need the number to distinguish
        if distribution_type == DistributionType.Weibull:
            params.append({"name": "beta", "default": 1, "location": 0, "bounds": (INFINITESIMAL, None)})
        else:
            params.append({"name": "beta", "default": 0, "location": 0, "bounds": (None, None)})
        params.append({"name": "eta", "default": 1, "location": 1, "bounds": (INFINITESIMAL, None)})
    elif ncomp > 1:
        for i in range(ncomp):
            # the shape params, beta and eta, of each distribution
            if distribution_type == DistributionType.Weibull:
                params.append({"name": "beta{0}".format(i+1), "default": 1, "location": i*2, "bounds": (INFINITESIMAL, None)})
            else:
                params.append({"name": "beta{0}".format(i+1), "default": 0, "location": i*2, "bounds": (None, None)})
            params.append({"name": "eta{0}".format(i+1), "default": 10, "location": i * 2 + 1, "bounds": (INFINITESIMAL, None)})
        for i in range(ncomp-1):
            # the fraction of each distribution
            params.append({"name": "f{0}".format(i+1), "default": 1/ncomp, "location": ncomp*2 + i, "bounds": (0, 1)})
    else:
        raise ValueError(ncomp)
    
    return params


def sort_params_by_location_in_place(params: list) -> list:
    params.sort(key=lambda element: element["location"])


def get_bounds(params: list):
    bounds = []
    for param in params:
        bounds.append(param["bounds"])
    return bounds


def get_constrains(ncomp: int):
    cons = [{'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-ncomp:]) + INFINITESIMAL}]
    return cons


def get_defaults(params: list):
    defaults = []
    for param in params:
        defaults.append(param["default"])
    return defaults


def get_lambda_string(ncomp:int, params: list, distribution_type: DistributionType) -> str:
    if ncomp == 1:
        if distribution_type == DistributionType.Weibull:
            return "lambda x, beta, eta: weibull(x, beta, eta)"
        else:
            return "lambda x, beta, eta: normal(x, beta, eta)"
    elif ncomp > 1:
        parameter_list = ", ".join(["x"] + [param["name"] for param in params])
        # " + " to connect each sub-function
        # the left list `["f{0}*{function_name}(x, beta{0}, eta{0})".format(i+1) for i in range(ncomp-1)]` means the n-1 sub-functions
        # the right list which only contains one element is the last sub-function
        if distribution_type == DistributionType.Weibull:
            expression = " + ".join(["f{0}*weibull(x, beta{0}, eta{0})".format(i+1) for i in range(ncomp-1)] + ["({0})*weibull(x, beta{1}, eta{1})".format("-".join(["1"]+["f{0}".format(i+1) for i in range(ncomp-1)]), ncomp)])
        else:
            expression = " + ".join(["f{0}*normal(x, beta{0}, eta{0})".format(i+1) for i in range(ncomp-1)] + ["({0})*normal(x, beta{1}, eta{1})".format("-".join(["1"]+["f{0}".format(i+1) for i in range(ncomp-1)]), ncomp)])
        lambda_string = "lambda {0}: {1}".format(parameter_list, expression)
        return lambda_string
    else:
        raise ValueError(ncomp)


# prcess the raw params list to make it easy to use
def process_params(ncomp: int, func_params: list, fitted_params: list, distribution_type: DistributionType) -> list:
    if ncomp == 1:
        assert len(fitted_params) == 2
        return [tuple(fitted_params)]
    elif ncomp > 1:
        # initialize the result list
        processed = []
        for i in range(ncomp):
            processed.append([None, None, 1])
        
        for func_param in func_params:
            name = func_param["name"] # type: str
            if name.startswith("beta"):
                comp_index = int(name[4:]) - 1
                processed[comp_index][0] = fitted_params[func_param["location"]]
            elif name.startswith("eta"):
                comp_index = int(name[3:]) - 1
                processed[comp_index][1] = fitted_params[func_param["location"]]
            elif name.startswith("f"):
                comp_index = int(name[1:]) - 1
                processed[comp_index][2] = fitted_params[func_param["location"]]
                processed[-1][2] -= fitted_params[func_param["location"]]
            else:
                raise ValueError(func_param)
        
        # sort the list by mean
        if distribution_type == DistributionType.Weibull:
            processed.sort(key=lambda element: weibull_mean(*element[:-1]))
        else:
            processed.sort(key=lambda element: normal_mean(*element[:-1]))
        return processed
    else:
        raise ValueError(ncomp)


# The pdf function of Weibull distribution
def weibull(x, beta, eta):
    return beta/eta*(x / eta)**(beta-1) * np.exp(-(x/eta)**beta)

# The mixed pdf function of two Weibull distribution
def double_weibull(x, beta1, eta1, beta2, eta2, f):
    return f*weibull(x, beta1, eta1) + (1-f)*weibull(x, beta2, eta2)

# The mixed pdf function of three Weibull distribution
def triple_weibull(x, beta1, eta1, beta2, eta2, beta3, eta3, f1, f2):
    return f1*weibull(x, beta1, eta1) + f2*weibull(x, beta2, eta2) + (1-f1-f2)*weibull(x, beta3, eta3)

# The mixed pdf function of four Weibull distribution
def quadruple_weibull(x, beta1, eta1, beta2, eta2, beta3, eta3, beta4, eta4, f1, f2, f3):
    return f1*weibull(x, beta1, eta1) + f2*weibull(x, beta2, eta2) + f3*weibull(x, beta3, eta3) + (1-f1-f2-f3)*weibull(x, beta4, eta4)

# call this func to get the mixed weibull function and related data
def get_mixed_weibull(ncomp) -> (callable, list, list, list, list):
    check_ncomp(ncomp)
    local_params = {"__tempmMixedFunc": None}
    func_params = get_params(ncomp, DistributionType.Weibull)
    sort_params_by_location_in_place(func_params)
    lambda_string = get_lambda_string(ncomp, func_params, DistributionType.Weibull)
    bounds = get_bounds(func_params)
    constrains = get_constrains(ncomp)
    defaults = get_defaults(func_params)
    exec("__tempmMixedFunc=" + lambda_string, None, local_params)
    return local_params["__tempmMixedFunc"], bounds, constrains, defaults, func_params

def weibull_mean(beta, eta):
    return eta*gamma(1/beta+1)

def weibull_median(beta, eta):
    return eta*(np.log(2)**(1/beta))

def weibull_mode(beta, eta):
    return eta*(1-1/beta)**(1/beta)

def weibull_std_deviation(beta, eta):
    return eta*np.sqrt(gamma(2/beta+1) - gamma(1/beta+1)**2)

def weibull_variance(beta, eta):
    return (eta**2)*(gamma(2/beta+1)-gamma(1/beta+1)**2)

def weibull_skewness(beta, eta):
    return (2*gamma(1/beta+1)**3 - 3*gamma(2/beta+1)*gamma(1/beta+1) + gamma(3/beta+1)) / (gamma(2/beta+1)-gamma(1/beta+1)**2)**(3/2)

def weibull_kurtosis(beta, eta):
    return (-3*gamma(1/beta+1)**4 + 6*gamma(2/beta+1)*gamma(1/beta+1)**2 - 4*gamma(3/beta+1)*gamma(1/beta+1) + gamma(4/beta+1)) / (gamma(2/beta+1)-gamma(1/beta+1)**2)**2




def normal(x, beta, eta):
    # return (1 / (x*eta*np.sqrt(2*np.pi))) * np.exp(-np.square((np.log(x)-beta) / eta) / 2)
    return (1 / np.sqrt(2*np.pi*eta))*np.exp(-(x-beta)**2/(2*eta**2))

def get_mixed_normal(ncomp) -> (callable, list, list, list, list):
    check_ncomp(ncomp)
    local_params = {"__tempmMixedFunc": None}
    func_params = get_params(ncomp, DistributionType.LogNormal)
    sort_params_by_location_in_place(func_params)
    lambda_string = get_lambda_string(ncomp, func_params, DistributionType.LogNormal)
    bounds = get_bounds(func_params)
    constrains = get_constrains(ncomp)
    defaults = get_defaults(func_params)
    exec("__tempmMixedFunc=" + lambda_string, None, local_params)
    return local_params["__tempmMixedFunc"], bounds, constrains, defaults, func_params

def normal_mean(beta, eta):
    return beta
