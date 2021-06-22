__all__ = ["ComponentSetting", "RandomSetting"]

import typing

import numpy as np
from scipy.stats import truncnorm

class ComponentSetting:
    SHAPE_RANGE = (-100.0, 100.0)
    LOC_RANGE = (-15.0, 15.0)
    SCALE_RANGE = (1E-4, 100.0)  # CAN NOT EQUAL TO ZERO
    WEIGHT_RANGE = (1.0, 100.0)

    def __init__(self, shape, loc, scale, weight):
        self.__shape_mean, self.__shape_std = shape
        self.__loc_mean, self.__loc_std = loc
        self.__scale_mean,  self.__scale_std = scale
        self.__weight_mean, self.__weight_std = weight

    @property
    def shape(self):
        return self.__shape_mean, self.__shape_std

    @property
    def loc(self):
        return self.__loc_mean, self.__loc_std

    @property
    def scale(self):
        return self.__scale_mean, self.__scale_std

    @property
    def weight(self):
        return self.__weight_mean, self.__weight_std

    def get_mean_params(self) -> np.ndarray:
        return np.array([self.__shape_mean, self.__loc_mean, self.__scale_mean, self.__weight_mean])

    def get_random_params(self, n_samples=1) -> typing.List[np.ndarray]:
        params = [self.shape, self.loc, self.scale, self.weight]
        ranges = [self.SHAPE_RANGE, self.LOC_RANGE, self.SCALE_RANGE, self.WEIGHT_RANGE]

        params_array = []
        for (mean, std), (minimum, maximum) in zip(params, ranges):
            if std == 0.0:
                params_array.append(np.full(shape=n_samples, fill_value=mean, dtype=np.float64))
            else:
                a = (minimum - mean) / std
                b = (maximum - mean) / std
                random_params = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples)
                params_array.append(random_params)
        # a list of shape(n_samples) array
        # [shape, loc, scale, weight]
        return params_array

class RandomSetting:
    def __init__(self, target: typing.List[dict]):
        self.__components = [ComponentSetting(**params) for params in target]

    @property
    def n_components(self) -> int:
        return len(self.__components)

    @property
    def mean_param(self):
        res = np.full(self.n_components*4, fill_value=np.nan)
        for i, comp in enumerate(self.__components):
            mean_params = comp.get_mean_params()
            res[i*3:(i+1)*3] = mean_params[:-1]
            res[-self.n_components+i] = mean_params[-1]
        return res

    def get_random_params(self, n_samples=1) -> np.ndarray:
        res = np.full((n_samples, self.n_components*4), fill_value=np.nan)
        for i, comp in enumerate(self.__components):
            rand_params = comp.get_random_params(n_samples=n_samples)
            res[:, i*3] = rand_params[0]
            res[:, i*3+1] = rand_params[1]
            res[:, i*3+2] = rand_params[2]
            res[:, -self.n_components+i] = rand_params[-1]
        return res
