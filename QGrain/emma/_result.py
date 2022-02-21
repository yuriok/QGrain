import copy
import typing

import numpy as np

from ..model import GrainSizeDataset
from ..ssu import get_distance_func_by_name
from ._kernel import KernelType
from ._setting import EMMAResolverSetting


class EMMAResult:
    def __init__(self, dataset: GrainSizeDataset,
                 kernel_type: KernelType,
                 n_members: int,
                 resolver_setting: EMMAResolverSetting,
                 proportions: np.ndarray,
                 end_members: np.ndarray,
                 time_spent: float,
                 history: typing.List[typing.Tuple[np.ndarray, np.ndarray]]):
        self.__dataset = dataset
        self.__kernel_type = kernel_type
        self.__n_members = n_members
        self.__resolver_setting = resolver_setting
        self.__proportions = proportions
        self.__end_members = end_members
        self.__X_hat = proportions @ end_members
        self.__time_spent = time_spent
        self.__history = history

    @property
    def dataset(self) -> GrainSizeDataset:
        return self.__dataset

    @property
    def kernel_type(self) -> KernelType:
        return self.__kernel_type

    @property
    def n_samples(self) -> int:
        return self.__dataset.n_samples

    @property
    def n_members(self) -> int:
        return self.__n_members

    @property
    def n_classes(self) -> int:
        return len(self.__dataset.classes_φ)

    @property
    def resolver_setting(self) -> EMMAResolverSetting:
        return self.__resolver_setting

    @property
    def proportions(self) -> np.ndarray:
        return self.__proportions

    @property
    def end_members(self) -> np.ndarray:
        return self.__end_members

    @property
    def X_hat(self) -> np.ndarray:
        return self.__X_hat

    @property
    def time_spent(self) -> float:
        return self.__time_spent

    @property
    def n_iterations(self) -> int:
        return len(self.__history)

    def get_distance(self, distance: str):
        distance_func = get_distance_func_by_name(distance)
        return distance_func(self.__X_hat, self.__dataset.distributions)

    def get_distance_series(self, distance: str):
        distance_series = []
        distance_func = get_distance_func_by_name(distance)
        X = self.__dataset.distributions
        for fractions, end_members in self.__history:
            X_hat = fractions @ end_members
            distance = distance_func(X_hat, X)
            distance_series.append(distance)
        return distance_series

    def get_class_wise_distance_series(self, distance: str):
        X_hat = self.__X_hat
        X = self.__dataset.distributions
        distance_func = get_distance_func_by_name(distance)
        series = [distance_func(X_hat[:, i], X[:, i]) for i in range(self.n_classes)]
        return series

    def get_sample_wise_distance_series(self, distance: str):
        X_hat = self.__X_hat
        X = self.__dataset.distributions
        distance_func = get_distance_func_by_name(distance)
        series = [distance_func(X_hat[i, :], X[i, :]) for i in range(self.n_samples)]
        return series

    @property
    def history(self):
        for fractions, end_members in self.__history:
            copy_result = copy.copy(self)
            copy_result.__proportions = fractions
            copy_result.__end_members = end_members
            copy_result.__X_hat = fractions @ end_members
            yield copy_result
