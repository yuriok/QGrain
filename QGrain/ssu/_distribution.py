import typing
from enum import Enum, unique

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm, weibull_min

_INFINITESIMAL = 1e-8


@unique
class DistributionType(Enum):
    Normal = "Normal"
    SkewNormal = "Skew Normal"
    Weibull = "Weibull"
    GeneralWeibull = "General Weibull"


def relu(x):
    return np.maximum(x, 1e-8)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


class Normal:
    NAME = "Normal"
    N_PARAMETERS = 2
    PARAMETER_NAMES = ("Location", "Scale")
    PARAMETER_BOUNDS = ((None, None),
                        (_INFINITESIMAL, None))

    @staticmethod
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: np.ndarray):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, Normal.N_PARAMETERS+1, n_components)
        locations = np.expand_dims(parameters[:, 0, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 1, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 2, :], axis=1), 1)
        components = norm.pdf(classes, loc=locations, scale=scales) * interval
        mvsk = norm.stats(loc=locations[:, :, 0],scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, mvsk

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((Normal.N_PARAMETERS+1, n_components))
        defaults[0] = np.random.random((n_components,)) * 2.0 + 5.0
        defaults[1] = np.random.random((n_components,)) * 0.1 + 2.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults


class SkewNormal:
    NAME = "Skew Normal"
    N_PARAMETERS = 3
    PARAMETER_NAMES = ("Shape", "Location", "Scale")
    PARAMETER_BOUNDS = ((None, None),
                        (None, None),
                        (_INFINITESIMAL, None))

    @staticmethod
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: np.ndarray):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, SkewNormal.N_PARAMETERS+1, n_components)
        shapes = np.expand_dims(parameters[:, 0, :], 2).repeat(n_classes, 2)
        locations = np.expand_dims(parameters[:, 1, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 2, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 3, :], axis=1), 1)
        components = skewnorm.pdf(classes, shapes, loc=locations, scale=scales) * interval
        mvsk = skewnorm.stats(shapes[:, :, 0], loc=locations[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, mvsk

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((SkewNormal.N_PARAMETERS+1, n_components))
        defaults[0] = np.random.random((n_components,)) * 0.1
        defaults[1] = np.random.random((n_components,)) * 2.0 + 5.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 2.0
        defaults[3] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults


class Weibull:
    NAME = "Weibull"
    N_PARAMETERS = 2
    PARAMETER_NAMES = ("Shape", "Scale")
    PARAMETER_BOUNDS = ((_INFINITESIMAL, None),
                        (_INFINITESIMAL, None))

    @staticmethod
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: np.ndarray):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, Weibull.N_PARAMETERS+1, n_components)
        shapes = np.expand_dims(relu(parameters[:, 0, :]), 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 1, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 2, :], axis=1), 1)
        components = weibull_min.pdf(classes, shapes, scale=scales) * interval
        mvsk = weibull_min.stats(shapes[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, mvsk

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((Weibull.N_PARAMETERS+1, n_components))
        defaults[0] = np.random.random((n_components,)) * 0.1 + 3.60234942
        defaults[1] = np.random.random((n_components,)) * 0.1 + 3.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults


class GeneralWeibull:
    NAME = "General Weibull"
    N_PARAMETERS = 3
    PARAMETER_NAMES = ("Shape", "Location", "Scale")
    PARAMETER_BOUNDS = ((_INFINITESIMAL, None),
                        (None, None),
                        (_INFINITESIMAL, None))

    @staticmethod
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: np.ndarray):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, GeneralWeibull.N_PARAMETERS+1, n_components)
        shapes = np.expand_dims(relu(parameters[:, 0, :]), 2).repeat(n_classes, 2)
        locations = np.expand_dims(parameters[:, 1, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 2, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 3, :], axis=1), 1)
        components = weibull_min.pdf(classes, shapes, loc=locations, scale=scales) * interval
        mvsk = weibull_min.stats(shapes[:, :, 0], loc=locations[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, mvsk

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((GeneralWeibull.N_PARAMETERS+1, n_components))
        defaults[0] = np.random.random((n_components,)) * 0.1 + 3.60234942
        defaults[1] = np.random.random((n_components,)) * 2.0 + 5.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 3.0
        defaults[3] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults


def get_distribution(distribution_type: DistributionType):
    if distribution_type == DistributionType.Normal:
        return Normal
    elif distribution_type == DistributionType.SkewNormal:
        return SkewNormal
    elif distribution_type == DistributionType.Weibull:
        return Weibull
    elif distribution_type == DistributionType.GeneralWeibull:
        return GeneralWeibull
    else:
        raise NotImplementedError(distribution_type)


def get_sorted_indexes(
        distribution_type: DistributionType,
        parameters: np.ndarray,
        classes: np.ndarray,
        interval: float) -> typing.Tuple[int]:
    distribution_class = get_distribution(distribution_type)
    proportions, components, (m, v, s, k) = distribution_class.interpret(parameters, classes, interval)
    mean_values = [(i, mean) for i, mean in enumerate(np.median(m, axis=0))]
    # sort them by mean size
    mean_values.sort(key=lambda x: x[1], reverse=True)
    sorted_indexes = tuple([i for i, _ in mean_values])
    return sorted_indexes

def sort_parameters(
        distribution_type: DistributionType,
        parameters: np.ndarray,
        classes: np.ndarray,
        interval: float) -> np.ndarray:
    sorted_indexes = get_sorted_indexes(distribution_type, parameters, classes, interval)
    sorted_parameters = np.zeros_like(parameters)
    for i, j in enumerate(sorted_indexes):
        sorted_parameters[:, :, i] = parameters[:, :, j]
    return sorted_parameters
