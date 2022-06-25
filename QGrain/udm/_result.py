from __future__ import annotations

import copy
import typing

import numpy as np

from ..emma import KernelType
from ..model import GrainSizeDataset
from ..ssu import (DISTRIBUTION_CLASS_MAP, DistributionType, GeneralWeibull,
                   Normal, SkewNormal, Weibull, get_distance_function)
from ._setting import UDMAlgorithmSetting


class UDMResult:
    def __init__(self,
                 dataset: GrainSizeDataset,
                 kernel_type: KernelType,
                 n_components: int,
                 initial_parameters: np.ndarray,
                 resolver_setting: UDMAlgorithmSetting,
                 distribution_loss_series: typing.Iterable[float],
                 component_loss_series: typing.Iterable[float],
                 time_spent: float,
                 final_parameters: np.ndarray,
                 history: typing.Iterable[np.ndarray]):
        self.__dataset = dataset
        self.__kernel_type = kernel_type
        self.__n_components = n_components
        self.__initial_parameters = initial_parameters
        self.__resolver_setting = resolver_setting
        self.__distribution_loss_series = distribution_loss_series
        self.__component_loss_series = component_loss_series
        self.__time_spent = time_spent
        self.__final_parameters = final_parameters
        self.__history = [final_parameters] if history is None else history
        self.__classes = np.expand_dims(np.expand_dims(self.dataset.classes_φ, axis=0), axis=0).repeat(self.n_samples, axis=0).repeat(self.n_components, axis=1)
        self.__interval = np.abs((self.dataset.classes_φ[0]-self.dataset.classes_φ[-1]) / (self.n_classes-1))
        self.__distribution_class = DISTRIBUTION_CLASS_MAP[DistributionType.__members__[self.kernel_type.name]]
        self.update(final_parameters)

    @property
    def dataset(self) -> GrainSizeDataset:
        return self.__dataset

    @property
    def n_samples(self) -> int:
        return self.dataset.n_samples

    @property
    def n_classes(self) -> int:
        return len(self.dataset.classes_μm)

    @property
    def kernel_type(self) -> KernelType:
        return self.__kernel_type

    @property
    def n_components(self) -> int:
        return self.__n_components

    @property
    def initial_parameters(self) -> np.ndarray:
        return self.__initial_parameters

    @property
    def resolver_setting(self) -> UDMAlgorithmSetting:
        return self.__resolver_setting

    @property
    def proportions(self) -> np.ndarray:
        return self.__proportions

    @property
    def components(self) -> np.ndarray:
        return self.__components

    @property
    def distribution_loss_series(self) -> np.ndarray:
        return self.__distribution_loss_series

    @property
    def component_loss_series(self) -> np.ndarray:
        return self.__component_loss_series

    @property
    def final_parameters(self) -> np.ndarray:
        return self.__final_parameters

    @property
    def time_spent(self) -> float:
        return self.__time_spent

    @property
    def n_iterations(self) -> int:
        return len(self.__history)

    @property
    def history(self) -> typing.Iterable[UDMResult]:
        for parameters in self.__history:
            copy_result = copy.copy(self)
            self.update(parameters)
            yield copy_result

    def update(self, parameters: np.ndarray):
        proportions, components, mvsk = self.__distribution_class.interpret(parameters, self.__classes, self.__interval)
        self.__proportions = proportions
        self.__components = components

    def get_distance(self, distance: str) -> float:
        distance_func = get_distance_function(distance)
        predict = (self.proportions @ self.components).squeeze()
        return distance_func(predict, self.dataset.distribution_matrix)
