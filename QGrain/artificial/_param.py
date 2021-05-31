__all__ = ["ComponentParameter", "GenerateParameter"]

import typing

import numpy as np
from scipy.stats import skewnorm

class ComponentParameter:
    def __init__(self, shape: float, loc: float, scale: float, weight: float):
        self.__shape = shape
        self.__loc = loc
        self.__scale = scale
        self.__weight = weight
        mean, variance, skewness, kurtosis = skewnorm.stats(shape, loc=loc, scale=scale, moments="mvsk")
        std = np.sqrt(variance)
        median = skewnorm.median(shape, loc=loc, scale=scale)
        self.__moments = dict(mean=mean, median=median, std=std, skewness=skewness, kurtosis=kurtosis)

    @property
    def shape(self) -> float:
        return self.__shape

    @property
    def loc(self) -> float:
        return self.__loc

    @property
    def scale(self) -> float:
        return self.__scale

    @property
    def weight(self) -> float:
        return self.__weight

    @property
    def func_args(self) -> typing.Iterable[float]:
        return (self.shape, self.loc, self.scale)

    @property
    def moments(self) -> dict:
        return self.__moments

class GenerateParameter:
    def __init__(self, params: np.ndarray):
        self.__n_components, left = divmod(len(params), 4)
        assert left == 0
        self.__components = [ComponentParameter(*params[i*3:(i+1)*3], params[-self.n_components+i]) for i in range(self.__n_components)]
        self.__sum_weight = sum([component.weight for component in self.__components])
        self.__fractions = tuple([component.weight/self.__sum_weight for component in self.__components])

    @property
    def n_components(self) -> int:
        return self.__n_components

    @property
    def components(self) -> typing.List[ComponentParameter]:
        return self.__components

    @property
    def sum_weight(self) -> float:
        return self.__sum_weight

    @property
    def fractions(self) -> typing.Tuple[float]:
        return self.__fractions
