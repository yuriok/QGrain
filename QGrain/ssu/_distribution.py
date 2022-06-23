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
    N_PARAMS = 2
    PARAM_NAMES = ("Location", "Scale")
    PARAM_BOUNDS = ((None, None),
                    (_INFINITESIMAL, None))

    @staticmethod
    def interpret(params: np.ndarray, classes: np.ndarray, interval: np.ndarray):
        n_samples, n_components, n_classes = classes.shape
        # params = params.reshape((n_samples, Normal.N_PARAMS+1, n_components))
        assert params.ndim == 3
        assert params.shape == (n_samples, Normal.N_PARAMS+1, n_components)
        locations = np.expand_dims(params[:, 0, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(params[:, 1, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(params[:, 2, :], axis=1), 1)
        components = norm.pdf(classes, loc=locations, scale=scales) * interval
        mvsk = norm.stats(loc=locations[:, :, 0],scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, mvsk

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((Normal.N_PARAMS+1, n_components))
        defaults[0] = np.random.random((n_components,)) * 2.0 + 5.0
        defaults[1] = np.random.random((n_components,)) * 0.1 + 2.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults

    @staticmethod
    def get_initial_guess(mean: float, std: float, skewness: float=0.0) -> np.ndarray:
        return np.array([mean, std])


class SkewNormal:
    NAME = "Skew Normal"
    N_PARAMS = 3
    PARAM_NAMES = ("Shape", "Location", "Scale")
    PARAM_BOUNDS = ((None, None),
                    (None, None),
                    (_INFINITESIMAL, None))

    @staticmethod
    def interpret(params: np.ndarray, classes: np.ndarray, interval: np.ndarray):
        n_samples, n_components, n_classes = classes.shape
        # params = params.reshape((n_samples, SkewNormal.N_PARAMS+1, n_components))
        assert params.ndim == 3
        assert params.shape == (n_samples, SkewNormal.N_PARAMS+1, n_components)
        shapes = np.expand_dims(params[:, 0, :], 2).repeat(n_classes, 2)
        locations = np.expand_dims(params[:, 1, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(params[:, 2, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(params[:, 3, :], axis=1), 1)
        components = skewnorm.pdf(classes, shapes, loc=locations, scale=scales) * interval
        mvsk = skewnorm.stats(shapes[:, :, 0], loc=locations[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, mvsk

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((SkewNormal.N_PARAMS+1, n_components))
        defaults[0] = np.random.random((n_components,)) * 0.1
        defaults[1] = np.random.random((n_components,)) * 2.0 + 5.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 2.0
        defaults[3] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults

    @staticmethod
    def get_initial_guess(mean: float, std: float, skewness: float=0.0) -> np.ndarray:
        x0 = [np.random.rand()*0.1, mean, std]
        def closure(args):
            m, v, s, k = skewnorm.stats(*args, moments="mvsk")
            errors = (mean-m)**2 + (std-np.sqrt(v))**2 + (skewness-k)**2
            return errors
        res = minimize(closure, x0=x0)
        return res.x


class Weibull:
    NAME = "Weibull"
    N_PARAMS = 2
    PARAM_NAMES = ("Shape", "Scale")
    PARAM_BOUNDS = ((_INFINITESIMAL, None),
                    (_INFINITESIMAL, None))

    @staticmethod
    def interpret(params: np.ndarray, classes: np.ndarray, interval: np.ndarray):
        n_samples, n_components, n_classes = classes.shape
        # params = params.reshape((n_samples, Weibull.N_PARAMS+1, n_components))
        assert params.ndim == 3
        assert params.shape == (n_samples, Weibull.N_PARAMS+1, n_components)
        shapes = np.expand_dims(relu(params[:, 0, :]), 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(params[:, 1, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(params[:, 2, :], axis=1), 1)
        components = weibull_min.pdf(classes, shapes, scale=scales) * interval
        mvsk = weibull_min.stats(shapes[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, mvsk

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((Weibull.N_PARAMS+1, n_components))
        defaults[0] = np.random.random((n_components,)) * 0.1 + 3.60234942
        defaults[1] = np.random.random((n_components,)) * 0.1 + 3.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults

    @staticmethod
    def get_initial_guess(mean: float, std: float, skewness: float=0.0) -> np.ndarray:
        x0 = [np.random.rand()*0.1+3.0, np.random.rand()*0.1+3.0]
        def closure(args):
            m, v, s, k = weibull_min.stats(args[0], scale=args[1], moments="mvsk")
            errors = (mean-m)**2 + (std-np.sqrt(v))**2 + (skewness-k)**2
            return errors
        res = minimize(closure, x0=x0)
        return res.x


class GeneralWeibull:
    NAME = "General Weibull"
    N_PARAMS = 3
    PARAM_NAMES = ("Shape", "Location", "Scale")
    PARAM_BOUNDS = ((_INFINITESIMAL, None),
                    (None, None),
                    (_INFINITESIMAL, None))

    @staticmethod
    def interpret(params: np.ndarray, classes: np.ndarray, interval: np.ndarray):
        n_samples, n_components, n_classes = classes.shape
        # params = params.reshape((n_samples, GeneralWeibull.N_PARAMS+1, n_components))
        assert params.ndim == 3
        assert params.shape == (n_samples, GeneralWeibull.N_PARAMS+1, n_components)
        shapes = np.expand_dims(relu(params[:, 0, :]), 2).repeat(n_classes, 2)
        locations = np.expand_dims(params[:, 1, :], 2).repeat(n_classes, 2)
        scales = np.expand_dims(relu(params[:, 2, :]), 2).repeat(n_classes, 2)
        proportions = np.expand_dims(softmax(params[:, 3, :], axis=1), 1)
        components = weibull_min.pdf(classes, shapes, loc=locations, scale=scales) * interval
        mvsk = weibull_min.stats(shapes[:, :, 0], loc=locations[:, :, 0], scale=scales[:, :, 0], moments="mvsk")
        return proportions, components, mvsk

    @staticmethod
    def get_defaults(n_components: int):
        defaults = np.zeros((GeneralWeibull.N_PARAMS+1, n_components))
        defaults[0] = np.random.random((n_components,)) * 0.1 + 3.60234942
        defaults[1] = np.random.random((n_components,)) * 2.0 + 5.0
        defaults[2] = np.random.random((n_components,)) * 0.1 + 3.0
        defaults[3] = np.random.random((n_components,)) * 0.1 + 2.0
        return defaults

    @staticmethod
    def get_initial_guess(mean: float, std: float, skewness: float=0.0) -> np.ndarray:
        x0 = [np.random.rand()*0.1+3.0, np.random.rand()*0.1+mean-2.0, np.random.rand()*0.1+3.0]
        def closure(args):
            m, v, s, k = weibull_min.stats(*args, moments="mvsk")
            errors = (mean-m)**2 + (std-np.sqrt(v))**2 + (skewness-k)**2
            return errors
        res = minimize(closure, x0=x0)
        return res.x


DISTRIBUTION_CLASS_MAP = {
    DistributionType.Normal: Normal,
    DistributionType.SkewNormal: SkewNormal,
    DistributionType.Weibull: Weibull,
    DistributionType.GeneralWeibull: GeneralWeibull}
