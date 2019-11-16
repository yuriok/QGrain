import numpy as np
from scipy.special import gamma


INFINITESIMAL = 1e-100


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


def _check_ncomp(ncomp: int):
    # Check the validity of `ncomp`
    if type(ncomp) != int:
        raise TypeError(ncomp)
    if ncomp <= 0:
        raise ValueError(ncomp)


def _get_params(ncomp: int) -> list:
    params = []
    if ncomp == 1:
        # if there is only one component, the fraction is not needful
        # beta and eta also don't need the number to distinguish
        params.append({"name": "beta", "default": 1, "location": 0, "bounds": (INFINITESIMAL, None)})
        params.append({"name": "eta", "default": 1, "location": 1, "bounds": (INFINITESIMAL, None)})
    elif ncomp > 1:
        for i in range(ncomp):
            # the shape params, beta and eta, of each weibull distribution
            params.append({"name": "beta{0}".format(i+1), "default": 1, "location": i*2, "bounds": (INFINITESIMAL, None)})
            params.append({"name": "eta{0}".format(i+1), "default": 1, "location": i * 2 + 1, "bounds": (INFINITESIMAL, None)})
        for i in range(ncomp-1):
            # the fraction of each weibull distribution
            params.append({"name": "f{0}".format(i+1), "default": 1/ncomp, "location": ncomp*2 + i, "bounds": (0, 1)})
    else:
        raise ValueError(ncomp)
    
    return params


def _sort_params_by_location_in_place(params: list) -> list:
    return params.sort(key=lambda element: element["location"])


def _get_bounds(params: list):
    bounds = []
    for param in params:
        bounds.append(param["bounds"])
    return bounds


def _get_constrains(ncomp: int):
    cons = ({'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-ncomp:]) + INFINITESIMAL})
    return cons


def _get_defaults(params: list):
    defaults = []
    for param in params:
        defaults.append(param["default"])
    return defaults


def _get_lambda_string(ncomp:int, params) -> str:
    if ncomp == 1:
        return "lambda x, beta, eta: beta/eta*(x / eta)**(beta-1) * np.exp(-(x/eta)**beta)"
    elif ncomp > 1:
        parameter_list = ", ".join(["x"] + [param["name"] for param in params])
        # " + " to connect each `weibull` sub-function
        # the left list `["f{0}*weibull(x, beta{0}, eta{0})".format(i+1) for i in range(ncomp-1)]` means the n-1 sub-functions
        # the right list which only contains one element is the last sub-function
        expression = " + ".join(["f{0}*weibull(x, beta{0}, eta{0})".format(i+1) for i in range(ncomp-1)] + ["({0})*weibull(x, beta{1}, eta{1})".format("-".join(["1"]+["f{0}".format(i+1) for i in range(ncomp-1)]), ncomp)])
        lambda_string = "lambda {0}: {1}".format(parameter_list, expression)
        return lambda_string
    else:
        raise ValueError(ncomp)


# call this func to get the mixed weibull function and related data
def get_mixed_weibull(ncomp) -> (callable, list, list, list):
    local_params = {"__tempmMixedFunc": None}
    func_params = _get_params(ncomp)
    _sort_params_by_location_in_place(func_params)
    lambda_string = _get_lambda_string(ncomp, func_params)
    bounds = _get_bounds(func_params)
    constrains = _get_constrains(ncomp)
    defaults = _get_defaults(func_params)
    exec("__tempmMixedFunc=" + lambda_string, None, local_params)
    return local_params["__tempmMixedFunc"], bounds, constrains, defaults


def mean(beta, eta):
    return eta*gamma(1/beta+1)


def median(beta, eta):
    return eta*(np.log(2)**(1/beta))
