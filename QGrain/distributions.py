__all__ = ["Normal", "SkewNormal", "Weibull", "GeneralWeibull",
           "get_distribution", "sort_components"]

import typing

import numpy as np
from scipy.special import softmax
from scipy.stats import norm, skewnorm, weibull_min

from .models import DistributionType

_INFINITESIMAL = 1e-8
_WEIBULL_SHAPE_WHEN_SKEWNESS_IS_ZERO = 3.6023494257189443


def relu(x):
    return np.maximum(x, _INFINITESIMAL)


class Normal:
    NAME = "Normal"
    N_PARAMETERS = 2
    PARAMETER_NAMES = ("Location", "Scale")
    PARAMETER_BOUNDS = ((None, None),
                        (_INFINITESIMAL, None))

    @staticmethod
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: float):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, Normal.N_PARAMETERS + 1, n_components)
        locations = np.expand_dims(parameters[:, 0, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 1, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 2, :], axis=1), 1)
        components = norm.pdf(classes, loc=locations, scale=scales) * interval
        m, v, s, k = norm.stats(loc=locations[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, (m, np.sqrt(v), s, k)

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((Normal.N_PARAMETERS + 1, n_components))
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
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: float):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, SkewNormal.N_PARAMETERS + 1, n_components)
        shapes = np.expand_dims(parameters[:, 0, :], 2).repeat(n_classes, 2)
        locations = np.expand_dims(parameters[:, 1, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 2, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 3, :], axis=1), 1)
        components = skewnorm.pdf(classes, shapes, loc=locations, scale=scales) * interval
        m, v, s, k = skewnorm.stats(shapes[:, :, 0], loc=locations[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, (m, np.sqrt(v), s, k)

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((SkewNormal.N_PARAMETERS + 1, n_components))
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
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: float):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, Weibull.N_PARAMETERS + 1, n_components)
        shapes = np.expand_dims(relu(parameters[:, 0, :]), 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 1, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 2, :], axis=1), 1)
        components = weibull_min.pdf(classes, shapes, scale=scales) * interval
        m, v, s, k = weibull_min.stats(shapes[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, (m, np.sqrt(v), s, k)

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((Weibull.N_PARAMETERS + 1, n_components))
        defaults[0] = np.random.random((n_components,)) * 0.1 + 3.60234942
        defaults[1] = np.random.random((n_components,)) * 0.1 + 3.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults
    
    
class SymmetricWeibull:
    NAME = "Symmetric Weibull"
    N_PARAMETERS = 2
    PARAMETER_NAMES = ("Location", "Scale")
    PARAMETER_BOUNDS = ((None, None),
                        (_INFINITESIMAL, None))

    @staticmethod
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: float):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, SymmetricWeibull.N_PARAMETERS + 1, n_components)
        shapes = np.ones_like(classes) * _WEIBULL_SHAPE_WHEN_SKEWNESS_IS_ZERO
        locations = np.expand_dims(parameters[:, 0, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 1, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 2, :], axis=1), 1)
        components = weibull_min.pdf(classes, shapes, loc=locations, scale=scales) * interval
        m, v, s, k = weibull_min.stats(shapes[:, :, 0], loc=locations[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, (m, np.sqrt(v), s, k)

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((SymmetricWeibull.N_PARAMETERS + 1, n_components))
        defaults[0] = np.random.random((n_components,)) * 2.0 + 5.0
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
    def interpret(parameters: np.ndarray, classes: np.ndarray, interval: float):
        n_samples, n_components, n_classes = classes.shape
        assert parameters.ndim == 3
        assert parameters.shape == (n_samples, GeneralWeibull.N_PARAMETERS + 1, n_components)
        shapes = np.expand_dims(relu(parameters[:, 0, :]), 2).repeat(n_classes, 2)
        locations = np.expand_dims(parameters[:, 1, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(parameters[:, 2, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(parameters[:, 3, :], axis=1), 1)
        components = weibull_min.pdf(classes, shapes, loc=locations, scale=scales) * interval
        m, v, s, k = weibull_min.stats(shapes[:, :, 0], loc=locations[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, (m, np.sqrt(v), s, k)

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((GeneralWeibull.N_PARAMETERS + 1, n_components))
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
    elif distribution_type == DistributionType.SymmetricWeibull:
        return SymmetricWeibull
    elif distribution_type == DistributionType.GeneralWeibull:
        return GeneralWeibull
    else:
        raise NotImplementedError(distribution_type)


def _get_means(distribution_type: DistributionType, parameters: np.ndarray) -> np.ndarray:
    if distribution_type == DistributionType.Normal:
        m = norm.stats(loc=parameters[:, 0, :], scale=parameters[:, 1, :], moments="m")
    elif distribution_type == DistributionType.SkewNormal:
        m = skewnorm.stats(parameters[:, 0, :], loc=parameters[:, 1, :], scale=parameters[:, 2, :], moments="m")
    elif distribution_type == DistributionType.Weibull:
        m = weibull_min.stats(parameters[:, 0, :], scale=parameters[:, 1, :], moments="m")
    elif distribution_type == DistributionType.SymmetricWeibull:
        shapes = np.ones_like(parameters[:, 0, :]) * _WEIBULL_SHAPE_WHEN_SKEWNESS_IS_ZERO
        m = weibull_min.stats(shapes, loc=parameters[:, 0, :], scale=parameters[:, 1, :], moments="m")
    elif distribution_type == DistributionType.GeneralWeibull:
        m = weibull_min.stats(parameters[:, 0, :], loc=parameters[:, 1, :], scale=parameters[:, 2, :], moments="m")
    else:
        raise NotImplementedError(distribution_type)
    return m


def sort_components(distribution_type: DistributionType, parameters: np.ndarray, mode="last") -> np.ndarray:
    assert parameters.ndim == 3
    # modes = ["first", "last", "median", "respective"]
    sorted_parameters = np.zeros_like(parameters)
    sort_keys = -_get_means(distribution_type, parameters)
    if mode == "first":
        indexes = np.argsort(sort_keys[0], axis=-1)
        for i, j in enumerate(indexes):
            sorted_parameters[:, :, i] = parameters[:, :, j]
    elif mode == "last":
        indexes = np.argsort(sort_keys[-1], axis=-1)
        for i, j in enumerate(indexes):
            sorted_parameters[:, :, i] = parameters[:, :, j]
    elif mode == "median":
        indexes = np.argsort(np.median(sort_keys, axis=0), axis=-1)
        for i, j in enumerate(indexes):
            sorted_parameters[:, :, i] = parameters[:, :, j]
    elif mode == "respective":
        indexes = np.argsort(sort_keys)
        for i in range(parameters.shape[0]):
            for j in range(parameters.shape[2]):
                sorted_parameters[i, :, j] = parameters[i, :, indexes[i, j]]
    else:
        raise NotImplementedError(mode)
    return sorted_parameters
