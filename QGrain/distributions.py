from __future__ import annotations

__all__ = ["ComponentNumberInvalidError",
           "BaseDistribution",
           "normal_frequency",
           "NormalDistribution",
           "weibull_frequency",
           "WeibullDistribution",
           "skew_normal_frequency",
           "SkewNormalDistribution"]

import typing
from abc import ABC, abstractproperty, abstractstaticmethod
from threading import Lock

import numpy as np
from QGrain import DistributionType
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm, weibull_min


class ComponentNumberInvalidError(Exception):
    """
    Raises while the component number is invalid.
    """
    pass

class BaseDistribution(ABC):
    _INFINITESIMAL = 1e-2
    _CACHE_LOCK = Lock()
    _CACHE = {}

    def __init__(self, n_components: int):
        if not BaseDistribution.check_n_components(n_components):
            raise ComponentNumberInvalidError(n_components)
        self.__n_components = n_components

    @abstractstaticmethod
    def get_name() -> str:
        pass

    @abstractstaticmethod
    def get_type() -> DistributionType:
        pass

    @abstractstaticmethod
    def get_parameter_names() -> typing.Tuple[str]:
        pass

    @abstractstaticmethod
    def get_parameter_bounds() -> \
        typing.Tuple[
            typing.Tuple[
                typing.Union[None, int, float],
                typing.Union[None, int, float]]]:
        pass

    @abstractstaticmethod
    def get_reference_parameters(mean: float, std: float, skewness: float) -> np.ndarray:
        pass

    @abstractstaticmethod
    def get_moments(*args) -> dict:
        pass

    @property
    def bounds(self) -> \
        typing.Tuple[
            typing.Tuple[
                typing.Union[None, int, float],
                typing.Union[None, int, float]]]:
        bounds = []
        param_bounds = self.get_parameter_bounds()
        for i in range(self.n_components):
            for bound in param_bounds:
                bounds.append(bound)
        for i in range(self.n_components-1):
            bounds.append((0.0, 1.0))

        return tuple(bounds)

    @abstractproperty
    def defaults(self) -> typing.Tuple[float]:
        pass

    @property
    def constrains(self) -> typing.Tuple[typing.Dict]:
        if self.__n_components == 1:
            return ()
        else:
            return ({'type': 'ineq', 'fun': lambda args:  1 - sum(args[1-self.__n_components:]) + BaseDistribution._INFINITESIMAL})

    @abstractproperty
    def single_function(self) -> typing.Callable:
        pass

    @abstractproperty
    def mixed_function(self) -> typing.Callable:
        pass

    @property
    def parameter_count(self) -> int:
        return len(self.get_parameter_names())

    @property
    def n_components(self) -> int:
        return self.__n_components

    @property
    def total_parameter_count(self) -> int:
        return (self.parameter_count + 1) * self.n_components - 1

    @staticmethod
    def check_n_components(n_components: int) -> bool:
        if not isinstance(n_components, int):
            return False
        if n_components < 1:
            return False
        return True

    def unpack_parameters(self, fitted_parameters) -> typing.Tuple[typing.Tuple, float]:
        assert len(fitted_parameters) == self.total_parameter_count
        if self.n_components == 1:
            return ((tuple(fitted_parameters), 1.0),)
        else:
            expanded = list(fitted_parameters) + [1.0-sum(fitted_parameters[self.n_components*self.parameter_count:])]
            return tuple(((tuple(expanded[i*self.parameter_count:(i+1)*self.parameter_count]), expanded[self.n_components*self.parameter_count+i]) for i in range(self.n_components)))

    @staticmethod
    def get_lambda_string(base_function_name: str, n_components: int, parameter_names: typing.List[str]) -> str:
        if n_components == 1:
            parameter_string = ", ".join(["x"] + list(parameter_names))
            return f"lambda {parameter_string}: {base_function_name}({parameter_string})"
        else:
            parameter_string = ", ".join(["x"] + [f"{name}{i+1}" for i in range(n_components) for name in parameter_names] + [f"f{i+1}" for i in range(n_components-1)])
            # " + " to connect each sub-function
            # the previous sub-function str list means the m-1 sub-functions with n params `fj * base_func(x, param_1_j, ..., param_i_j, ..., param_n_j)`
            # the last sub-function str which represents `(1-f_1-...-f_j-...-f_m-1) * base_func(x, param_1_j, ..., param_i_j, ..., param_n_j)`
            previous_sub_function_strings = [f"f{i+1} * {base_function_name}(x, {', '.join([f'{name}{i+1}' for name in parameter_names])})" for i in range(n_components-1)]
            last_sub_function_string = f"({'-'.join(['1']+[f'f{i+1}' for i in range(n_components-1)])}) * {base_function_name}(x, {', '.join([f'{name}{n_components}' for name in parameter_names])})"
            lambda_string = f"lambda {parameter_string}: {' + '.join(previous_sub_function_strings + [last_sub_function_string])}"
            return lambda_string

    @staticmethod
    def get_initial_guess(distribution_type: DistributionType, reference: typing.Iterable[typing.Dict], fractions=None):
        parameters = []
        n_components = len(reference)
        distribution_class = None
        for sub_class in BaseDistribution.__subclasses__():
            if sub_class.get_type() == distribution_type:
                    distribution_class = sub_class
        if distribution_class is None:
            raise ValueError("There is no corresponding sub-class of this distribution type.")

        for component_ref in reference:
            component_params = distribution_class.get_reference_parameters(**component_ref)
            parameters.extend(component_params)
        if fractions is None:
            for i in range(n_components-1):
                parameters.append(1/n_components)
        else:
            assert len(fractions) == n_components
            parameters.extend(fractions[:-1])
        return np.array(parameters)

    @staticmethod
    def get_distribution(distribution_type: DistributionType, n_components: int) -> BaseDistribution:
        key = (distribution_type, n_components)
        distribution = None
        BaseDistribution._CACHE_LOCK.acquire()
        if key in BaseDistribution._CACHE.keys():
            distribution = BaseDistribution._CACHE[key]
        else:
            for sub_class in BaseDistribution.__subclasses__():
                if sub_class.get_type() == distribution_type:
                    distribution = sub_class(n_components)
                    BaseDistribution._CACHE[key] = distribution
        BaseDistribution._CACHE_LOCK.release()

        if distribution is None:
            raise ValueError("There is no corresponding sub-class of this distribution type.")
        else:
            return distribution

def normal_frequency(classes_φ, loc, scale):
    interval = abs((classes_φ[0]-classes_φ[-1]) / (len(classes_φ)-1))
    pdf = norm.pdf(classes_φ, loc=loc, scale=scale)
    frequency = pdf * interval
    return frequency

class NormalDistribution(BaseDistribution):
    @staticmethod
    def get_name() -> str:
        return "Normal"

    @staticmethod
    def get_type() -> DistributionType:
        return DistributionType.Normal

    @staticmethod
    def get_parameter_names() -> typing.Tuple[str]:
        return ("loc", "scale")

    @staticmethod
    def get_parameter_bounds() -> \
        typing.Tuple[
            typing.Tuple[
                typing.Union[None, int, float],
                typing.Union[None, int, float]]]:
        return ((None, None),
                (BaseDistribution._INFINITESIMAL, None))

    @property
    def defaults(self) -> typing.Tuple[float]:
        defaults = []
        for i in range(1, self.n_components+1):
            defaults.append(5 + i) # loc
            defaults.append(1 + 0.1*i) # scale
        for i in range(self.n_components-1):
            defaults.append(1.0 / self.n_components) # fraction
        return tuple(defaults)

    @property
    def single_function(self) -> typing.Callable:
        return normal_frequency

    @property
    def mixed_function(self) -> typing.Callable:
        lambda_string = self.get_lambda_string("normal_frequency", self.n_components, self.get_parameter_names())
        local_params = {"mixed_function": None}
        exec(f"mixed_function = {lambda_string}", None, local_params)
        return local_params["mixed_function"]

    @staticmethod
    def get_moments(*args) -> dict:
        assert len(args) == len(NormalDistribution.get_parameter_names())
        m, v, s, k = norm.stats(*args, moments="mvsk")
        std = np.sqrt(v)
        moments = dict(mean=m, std=std, skewness=s, kurtosis=k)
        return moments

    @staticmethod
    def get_reference_parameters(mean: float, std: float, skewness: float=0.0) -> np.ndarray:
        return np.array([mean, std])

def weibull_frequency(classes_φ, shape, loc, scale):
    interval = abs((classes_φ[0]-classes_φ[-1]) / (len(classes_φ)-1))
    pdf = weibull_min.pdf(classes_φ, shape, loc=loc, scale=scale)
    frequency = pdf * interval
    return frequency

class WeibullDistribution(BaseDistribution):
    @staticmethod
    def get_name() -> str:
        return "Weibull"

    @staticmethod
    def get_type() -> DistributionType:
        return DistributionType.Weibull

    @staticmethod
    def get_parameter_names() -> typing.Tuple[str]:
        return ("shape", "loc", "scale")

    @staticmethod
    def get_parameter_bounds() -> \
        typing.Tuple[
            typing.Tuple[
                typing.Union[None, int, float],
                typing.Union[None, int, float]]]:
        return ((BaseDistribution._INFINITESIMAL, None),
                (None, None),
                (BaseDistribution._INFINITESIMAL, None))

    @property
    def defaults(self) -> typing.Tuple[float]:
        defaults = []
        for i in range(1, self.n_components+1):
            defaults.append(3.60234942) # shape while skewness is 0
            defaults.append(5.0 + i)
            defaults.append(1.0 + 0.1*i)
        for i in range(self.n_components-1):
            defaults.append(1.0 / self.n_components)
        return tuple(defaults)

    @property
    def single_function(self) -> typing.Callable:
        return weibull_frequency

    @property
    def mixed_function(self) -> typing.Callable:
        lambda_string = self.get_lambda_string("weibull_frequency", self.n_components, self.get_parameter_names())
        local_params = {"mixed_function": None}
        exec(f"mixed_function = {lambda_string}", None, local_params)
        return local_params["mixed_function"]

    @staticmethod
    def get_moments(*args) -> dict:
        assert len(args) == len(WeibullDistribution.get_parameter_names())
        m, v, s, k = weibull_min.stats(*args, moments="mvsk")
        std = np.sqrt(v)
        moments = dict(mean=m, std=std, skewness=s, kurtosis=k)
        return moments

    @staticmethod
    def get_reference_parameters(mean: float, std: float, skewness: float) -> np.ndarray:
        shape_skew_0 = 3.60234942
        x0 = [shape_skew_0, 0.0, 1.0]
        target = {"mean": mean, "std": std, "skewness": skewness}

        def closure(args):
            current = WeibullDistribution.get_moments(*args)
            errors = sum([(current[key]-target[key])**2 for key in target.keys()])
            return errors

        res = minimize(closure, x0=x0,
                       bounds=WeibullDistribution.get_parameter_bounds(),
                       method="SLSQP",
                       options={"maxiter": 100, "ftol": 1e-6, "disp": False})
        return res.x

def skew_normal_frequency(classes_φ, shape, loc, scale):
    interval = abs((classes_φ[0]-classes_φ[-1]) / (len(classes_φ)-1))
    pdf = skewnorm.pdf(classes_φ, shape, loc=loc, scale=scale)
    frequency = pdf * interval
    return frequency

class SkewNormalDistribution(BaseDistribution):
    @staticmethod
    def get_name() -> str:
        return "Skew Normal"

    @staticmethod
    def get_type() -> DistributionType:
        return DistributionType.SkewNormal

    @staticmethod
    def get_parameter_names() -> typing.Tuple[str]:
        return ("shape", "loc", "scale")

    @staticmethod
    def get_parameter_bounds() -> \
        typing.Tuple[
            typing.Tuple[
                typing.Union[None, int, float],
                typing.Union[None, int, float]]]:
        return ((None, None),
                (None, None),
                (BaseDistribution._INFINITESIMAL, None))

    @property
    def defaults(self) -> typing.Tuple[float]:
        defaults = []
        for i in range(1, self.n_components+1):
            defaults.append(0.0) # shape while skewness is 0
            defaults.append(5 + i) # loc
            defaults.append(1 + 0.1*i) # scale
        for i in range(self.n_components-1):
            defaults.append(1.0 / self.n_components) # fraction
        return tuple(defaults)

    @property
    def single_function(self) -> typing.Callable:
        return skew_normal_frequency

    @property
    def mixed_function(self) -> typing.Callable:
        lambda_string = self.get_lambda_string("skew_normal_frequency", self.n_components, self.get_parameter_names())
        local_params = {"mixed_function": None}
        exec(f"mixed_function = {lambda_string}", None, local_params)
        return local_params["mixed_function"]

    @staticmethod
    def get_moments(*args) -> dict:
        assert len(args) == len(SkewNormalDistribution.get_parameter_names())
        m, v, s, k = skewnorm.stats(*args, moments="mvsk")
        std = np.sqrt(v)
        moments = dict(mean=m, std=std, skewness=s, kurtosis=k)
        return moments

    @staticmethod
    def get_reference_parameters(mean: float, std: float, skewness: float=0.0) -> np.ndarray:
        x0 = [np.random.rand()*0.1, mean, std]
        target = {"mean": mean, "std": std, "skewness": skewness}

        def closure(args):
            current = SkewNormalDistribution.get_moments(*args)
            errors = sum([(current[key]-target[key])**2 for key in target.keys()])
            return errors

        res = minimize(closure, x0=x0,
                       bounds=SkewNormalDistribution.get_parameter_bounds(),
                       method="SLSQP",
                       options={"maxiter": 100, "ftol": 1e-6, "disp": False})

        return res.x

def log10MSE_distance(values: np.ndarray, targets: np.ndarray) -> float:
    return np.log10(np.mean(np.square(values - targets)))

def MSE_distance(values: np.ndarray, targets: np.ndarray) -> float:
    return np.mean(np.square(values - targets))

def p_norm(values: np.ndarray, targets: np.ndarray, p=2) -> float:
    return np.sum(np.abs(values - targets) ** p) ** (1 / p)

def cosine_distance(values: np.ndarray, targets: np.ndarray) -> float:
    if np.all(np.equal(values, 0.0)) or np.all(np.equal(targets, 0.0)):
        return 1.0
    cosine = np.sum(values * targets) / (np.sqrt(np.sum(np.square(values))) * np.sqrt(np.sum(np.square(targets))))
    return abs(cosine)

def angular_distance(values: np.ndarray, targets: np.ndarray) -> float:
    cosine = cosine_distance(values, targets)
    angular = 2 * np.arccos(cosine) / np.pi
    return angular

def get_distance_func_by_name(distance: str):
    if distance[-4:] == "norm":
        p = int(distance[0])
        return lambda x, y: p_norm(x, y, p)
    elif distance == "MSE":
        return lambda x, y: MSE_distance(x, y)
    elif distance == "log10MSE":
        return lambda x, y: log10MSE_distance(x, y)
    elif distance == "cosine":
        return lambda x, y: cosine_distance(x, y)
    elif distance == "angular":
        return lambda x, y: angular_distance(x, y)
    else:
        raise NotImplementedError(distance)
