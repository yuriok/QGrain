import typing

import numpy as np

from ..emma import KernelType
from ..model import GrainSizeDataset
from ..ssu import (DISTRIBUTION_CLASS_MAP, DistributionType, GeneralWeibull,
                   Normal, SkewNormal, Weibull, get_distance_function)
from ._setting import UDMAlgorithmSetting


class UDMResult:
    def __init__(self, dataset: GrainSizeDataset,
                 kernel_type: KernelType,
                 n_components: int,
                 params: np.ndarray,
                 distribution_loss_series: typing.Iterable[float],
                 component_loss_series: typing.Iterable[float],
                 initial_params: np.ndarray = None,
                 resolver_setting: UDMAlgorithmSetting = None,
                 time_spent: float = None,
                 history_params: typing.Iterable[np.ndarray] = None):
        self.dataset = dataset
        self.kernel_type = kernel_type
        self.n_components = n_components
        self.params = params
        self.distribution_loss_series = distribution_loss_series
        self.component_loss_series = component_loss_series
        self.initial_params = initial_params
        self.resolver_setting = resolver_setting
        self.time_spent = time_spent
        self.history_params = [params] if history_params is None else history_params
        self._classes = np.expand_dims(np.expand_dims(self.dataset.classes_φ, axis=0), axis=0).repeat(self.n_samples, axis=0).repeat(self.n_components, axis=1)
        self._interval = np.abs((self.dataset.classes_φ[0]-self.dataset.classes_φ[-1]) / (self.n_classes-1))
        self._distribution_class = DISTRIBUTION_CLASS_MAP[DistributionType.__members__[self.kernel_type.name]]
        self.update(params)

    @property
    def n_samples(self) -> int:
        return self.dataset.n_samples

    @property
    def n_classes(self) -> int:
        return len(self.dataset.classes_μm)

    def update(self, params: np.ndarray):
        proportions, components, mvsk = self._distribution_class.interpret(params, self._classes, self._interval)
        self.proportions = proportions
        self.components = components

    def clear_history(self):
        self.history_params.clear()
        self.history_params.append(self.params)

    def get_distance(self, distance: str) -> float:
        distance_func = get_distance_function(distance)
        X_hat = (self.proportions @ self.components).squeeze()
        return distance_func(X_hat, self.dataset.distribution_matrix)
