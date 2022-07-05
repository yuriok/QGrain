from __future__ import annotations

import copy
import typing

import numpy as np

from ..model import GrainSizeDataset
from ..ssu import get_distance_function
from ._kernel import KernelType
from ._setting import EMMAAlgorithmSetting


class EMMAResult:
    def __init__(self, dataset: GrainSizeDataset,
                 kernel_type: KernelType,
                 n_members: int,
                 initial_parameters: np.ndarray,
                 resolver_setting: EMMAAlgorithmSetting,
                 proportions: np.ndarray,
                 end_members: np.ndarray,
                 time_spent: float,
                 history: typing.List[typing.Tuple[np.ndarray, np.ndarray]]):
        self.__dataset = dataset
        self.__kernel_type = kernel_type
        self.__n_members = n_members
        self.__initial_parameters = initial_parameters
        self.__resolver_setting = resolver_setting
        self.__proportions = proportions
        self.__end_members = end_members
        self.__time_spent = time_spent
        self.__history = history
        modes = [(i, dataset.classes_μm[np.unravel_index(np.argmax(end_members[i]), end_members[i].shape)]) for i in range(n_members)]
        modes.sort(key=lambda x: x[1])
        self.__sorted_indexes = (i for i, _ in modes)
        self._sort()

    @property
    def dataset(self) -> GrainSizeDataset:
        return self.__dataset

    @property
    def n_samples(self) -> int:
        return self.dataset.n_samples

    @property
    def n_classes(self) -> int:
        return len(self.__dataset.classes_φ)

    @property
    def kernel_type(self) -> KernelType:
        return self.__kernel_type

    @property
    def n_members(self) -> int:
        return self.__n_members

    @property
    def initial_parameters(self) -> int:
        return self.__initial_parameters

    @property
    def resolver_setting(self) -> EMMAAlgorithmSetting:
        return self.__resolver_setting

    @property
    def proportions(self) -> np.ndarray:
        return self.__proportions

    @property
    def end_members(self) -> np.ndarray:
        return self.__end_members

    @property
    def time_spent(self) -> float:
        return self.__time_spent

    @property
    def n_iterations(self) -> int:
        return len(self.__history)

    @property
    def history(self) -> typing.Iterable[EMMAResult]:
        for fractions, end_members in self.__history:
            copy_result = copy.copy(self)
            copy_result.__proportions = fractions
            copy_result.__end_members = end_members
            copy_result._sort()
            yield copy_result

    def _sort(self):
        proportions = np.zeros_like(self.__proportions)
        end_members = np.zeros_like(self.__end_members)
        for i, j in enumerate(self.__sorted_indexes):
            proportions[:, i] = self.__proportions[:, j]
            end_members[i, :] = self.__end_members[j, :]
        self.__proportions = proportions
        self.__end_members = end_members

    def get_distance(self, distance: str) -> float:
        distance_func = get_distance_function(distance)
        predict = self.__proportions @ self.__end_members
        return distance_func(predict, self.__dataset.distribution_matrix)

    def get_distance_series(self, distance: str) -> np.ndarray:
        distances = []
        distance_func = get_distance_function(distance)
        for fractions, end_members in self.__history:
            predict = fractions @ end_members
            distance = distance_func(predict, self.__dataset.distribution_matrix)
            distances.append(distance)
        distances = np.array(distances)
        return distances

    def get_class_wise_distances(self, distance: str) -> np.ndarray:
        distance_func = get_distance_function(distance)
        predict = self.__proportions @ self.__end_members
        distances = distance_func(predict, self.__dataset.distribution_matrix, axis=0)
        return distances

    def get_sample_wise_distances(self, distance: str) -> np.ndarray:
        distance_func = get_distance_function(distance)
        predict = self.__proportions @ self.__end_members
        distances = distance_func(predict, self.__dataset.distribution_matrix, axis=1)
        return distances
