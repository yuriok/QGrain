from __future__ import annotations

import copy
import typing

import numpy as np

from ..emma import KernelType
from ..models import Dataset
from ..ssu import DistributionType, SSUResult, SSUTask, get_distance_function, get_distribution, get_sorted_indexes
from ._setting import UDMAlgorithmSetting


class UDMResult:
    def __init__(
            self, dataset: Dataset,
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
        self.__classes = np.expand_dims(np.expand_dims(self.dataset.classes_phi, axis=0), axis=0).repeat(self.n_samples, axis=0).repeat(self.n_components, axis=1)
        self.__interval = np.abs((self.dataset.classes_phi[0] - self.dataset.classes_phi[-1]) / (self.n_classes - 1))
        self.__sorted_indexes = get_sorted_indexes(self.distribution_type, final_parameters, self.__classes, self.__interval)
        self.update(final_parameters)

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    @property
    def n_classes(self) -> int:
        return len(self.dataset.classes)

    @property
    def kernel_type(self) -> KernelType:
        return self.__kernel_type

    @property
    def distribution_type(self) -> DistributionType:
        return DistributionType._member_map_[self.kernel_type.name]

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
        sorted_parameters = np.zeros_like(parameters)
        for i, j in enumerate(self.__sorted_indexes):
            sorted_parameters[:, :, i] = parameters[:, :, j]
        distribution_class = get_distribution(DistributionType.__members__[self.kernel_type.name])
        proportions, components, mvsk = distribution_class.interpret(sorted_parameters, self.__classes, self.__interval)
        proportions[np.logical_or(np.isnan(proportions), np.isinf(proportions))] = 0.0
        components[np.logical_or(np.isnan(components), np.isinf(components))] = 0.0
        self.__proportions = proportions
        self.__components = components

    def get_distance(self, distance: str) -> float:
        distance_func = get_distance_function(distance)
        predict = (self.proportions @ self.components).squeeze()
        return distance_func(predict, self.dataset.distributions)

    def to_ssu_results(self, progress_callback: typing.Callable = None) -> typing.List[SSUResult]:
        results = []
        weight = np.ones((1, self.n_components))
        initial_parameters = np.concatenate([self.initial_parameters, weight], axis=0).astype(np.float64)
        time_spent = self.time_spent / self.n_samples
        for i in range(self.n_samples):
            sample = self.dataset.samples[i]
            task = SSUTask(
                sample,
                self.distribution_type,
                self.n_components,
                resolver_setting=None,
                initial_parameters=initial_parameters)
            parameters=np.expand_dims(self.final_parameters[i], axis=0)
            history = [np.expand_dims(self.__history[j][i], axis=0) for j in range(self.n_iterations)]
            result = SSUResult(task, parameters, history=history, time_spent=time_spent)
            results.append(result)
            if progress_callback is not None:
                progress_callback(i / self.n_samples)
        if progress_callback is not None:
            progress_callback(1.0)
        return results
