__all__ = ["UDMResult"]

import copy
from typing import *

import numpy as np
from numpy import ndarray

from ..distributions import DistributionType, get_distribution, get_sorted_indexes
from ..emma import KernelType
from ..metrics import loss_numpy
from ..models import ArtificialDataset
from ..models import Dataset


class UDMResult:
    def __init__(self, dataset: [ArtificialDataset, Dataset], kernel_type: KernelType, n_components: int,
                 parameters: ndarray, time_spent: Union[int, float], x0: ndarray = None,
                 history: Sequence[ndarray] = None, settings: Dict[str, Any] = None,
                 loss_series: Dict[str, ndarray] = None):
        self._dataset = dataset
        self._kernel_type = kernel_type
        self._n_components = n_components
        self._parameters = parameters
        self._time_spent = time_spent
        self._x0 = x0
        self._history = history
        self._settings = settings
        self._loss_series = loss_series
        self._classes_phi = np.expand_dims(np.expand_dims(self.dataset.classes_phi, axis=0), axis=0).repeat(
            self.n_samples, axis=0).repeat(self.n_components, axis=1)
        self._interval_phi = np.abs((self.dataset.classes_phi[0] - self.dataset.classes_phi[-1]) / (self.n_classes - 1))
        self._sorted_indexes = get_sorted_indexes(self.distribution_type, parameters,
                                                  self._classes_phi, self._interval_phi)
        self._update(parameters)

    @property
    def dataset(self) -> Union[ArtificialDataset, Dataset]:
        return self._dataset

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def n_classes(self) -> int:
        return len(self.dataset.classes)

    @property
    def kernel_type(self) -> KernelType:
        return self._kernel_type

    @property
    def distribution_type(self) -> DistributionType:
        return DistributionType.__members__[self._kernel_type.name]

    @property
    def proportions(self) -> ndarray:
        return self._proportions

    @property
    def components(self) -> ndarray:
        return self._components

    @property
    def time_spent(self) -> Union[int, float]:
        return self._time_spent

    @property
    def x0(self) -> Optional[ndarray]:
        return self._x0

    @property
    def n_iterations(self) -> int:
        return len(self._history)

    @property
    def parameters(self) -> ndarray:
        return self._parameters

    @property
    def history(self):
        for parameters in self._history:
            copy_result = copy.copy(self)
            copy_result._update(parameters)
            yield copy_result

    @property
    def settings(self) -> Optional[Dict[str, Any]]:
        if self._settings is None:
            return None
        else:
            return self._settings.copy()

    def _update(self, parameters: np.ndarray):
        sorted_parameters = np.zeros_like(parameters)
        for i, j in enumerate(self._sorted_indexes):
            sorted_parameters[:, :, i] = parameters[:, :, j]
        distribution_class = get_distribution(DistributionType.__members__[self.kernel_type.name])
        proportions, components, _ = distribution_class.interpret(sorted_parameters, self._classes_phi,
                                                                  self._interval_phi)
        proportions[np.logical_or(np.isnan(proportions), np.isinf(proportions))] = 0.0
        components[np.logical_or(np.isnan(components), np.isinf(components))] = 0.0
        self._proportions = proportions
        self._components = components

    def loss(self, name: str) -> float:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = (self._proportions @ self._components).squeeze(1)
        return loss_func(prediction, observation, None)

    def loss_series(self, name: str) -> ndarray:
        if self.n_iterations == 0:
            raise ValueError("No history record.")
        else:
            series = []
            loss_func = loss_numpy(name)
            observation = self._dataset.distributions
            for result in self.history:
                prediction = (result._proportions @ result._components).squeeze(1)
                loss = loss_func(prediction, observation, None)
                series.append(loss)
            series = np.array(series)
            return series

    def class_wise_losses(self, name: str) -> ndarray:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = (self._proportions @ self._components).squeeze(1)
        losses = loss_func(prediction, observation, 0)
        return losses

    def sample_wise_losses(self, name: str) -> ndarray:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = (self._proportions @ self._components).squeeze(1)
        losses = loss_func(prediction, observation, 1)
        return losses
