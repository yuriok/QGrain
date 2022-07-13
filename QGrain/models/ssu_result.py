__all__ = ["SSUResultComponent", "SSUResult"]

import copy
from typing import *

import numpy as np
from numpy import ndarray

from .artificial_dataset import ArtificialSample
from .dataset import Sample
from ..distributions import DistributionType, get_distribution
from ..metrics import loss_numpy
from ..statistics import interval_phi


class SSUResultComponent:
    __slots__ = ("_classes", "_classes_phi", "_distribution", "_proportion", "_moments", "_parameters")

    def __init__(self, classes: ndarray, classes_phi: ndarray, distribution: ndarray,
                 proportion: float, moments: Tuple[float, float, float, float]):
        self._classes = classes
        self._classes_phi = classes_phi
        self._distribution = distribution
        self._proportion = proportion
        m, std, s, k = moments
        self._moments = dict(mean=m, std=std, skewness=s, kurtosis=k)

    def __repr__(self):
        return f"C({self._moments['mean']:.2f}, {self._proportion:.2%})"

    @property
    def classes(self) -> ndarray:
        return self._classes

    @property
    def classes_phi(self) -> ndarray:
        return self._classes_phi

    @property
    def interval_phi(self) -> float:
        return interval_phi(self._classes_phi)

    @property
    def distribution(self) -> ndarray:
        return self._distribution

    @property
    def proportion(self) -> float:
        return self._proportion

    @property
    def moments(self) -> dict:
        return self._moments

    @property
    def mean(self) -> float:
        return self._moments["mean"]

    @property
    def sorting_coefficient(self) -> float:
        return self._moments["std"]

    @property
    def skewness(self) -> float:
        return self._moments["skewness"]

    @property
    def kurtosis(self) -> float:
        return self._moments["kurtosis"]


class SSUResult:
    """
    This class represents the SSU result of each sample.
    """

    def __init__(self, sample: Union[Sample, ArtificialSample], distribution_type: DistributionType,
                 x0: ndarray, parameters: ndarray, time_spent: float):
        self._sample = sample
        self._distribution_type = distribution_type
        self._x0 = x0
        self._parameters = parameters
        self._time_spent = time_spent

        n_iterations, n_parameters, n_components = parameters.shape
        classes = np.expand_dims(np.expand_dims(sample.classes_phi, 0), 0).repeat(n_components, 1)
        distribution_class = get_distribution(distribution_type)
        proportions, components, (m, v, s, k) = distribution_class.interpret(
            np.expand_dims(self._parameters[-1], 0), classes, self._sample.interval_phi)
        proportions, components, (m, std, s, k) = proportions[0], components[0], (m[0], np.std(v[0]), s[0], k[0])
        distribution = (proportions @ components)[0]
        proportions = proportions[0]
        self._distribution = distribution
        self._proportions = proportions
        self._components = components
        self._moments = (m, std, s, k)

    def __repr__(self):
        return f"SSUResult({self._sample.name}, {self._parameters.shape[2]}, {self._distribution_type.name})"

    def __len__(self):
        return len(self._components)

    def __iter__(self):
        for i in range(len(self._components)):
            yield self._get_component(i)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get_component(item)
        elif isinstance(item, slice):
            return [self._get_component(index) for index in np.arange(len(self._components))[item]]
        else:
            raise TypeError(f"Component indices must be integers or slices, not {type(item)}.")

    @property
    def name(self) -> str:
        return self._sample.name

    @property
    def sample(self) -> Sample:
        return self._sample

    @property
    def classes(self) -> ndarray:
        return self._sample.classes

    @property
    def classes_phi(self) -> ndarray:
        return self._sample.classes_phi

    @property
    def interval_phi(self) -> float:
        return interval_phi(self._sample.classes_phi)

    @property
    def distribution(self) -> ndarray:
        return self._distribution

    @property
    def distribution_type(self) -> DistributionType:
        return self._distribution_type

    @property
    def x0(self) -> ndarray:
        return self._x0

    @property
    def parameters(self) -> ndarray:
        return self._parameters

    @property
    def time_spent(self):
        return self._time_spent

    @property
    def n_iterations(self):
        return self._parameters.shape[0]

    @property
    def n_parameters(self) -> int:
        return self._parameters.shape[1]

    @property
    def n_components(self) -> int:
        return self._parameters.shape[2]

    @property
    def is_valid(self) -> bool:
        valid = True
        for values in [self._proportions, self._components, self._distribution, *self._moments]:
            if np.any(np.logical_or(np.isnan(values), np.isinf(values))):
                valid = False
                break
        return valid

    @property
    def history(self):
        n_iterations, n_parameters, n_components = self._parameters.shape
        classes = np.expand_dims(np.expand_dims(
            self._sample.classes_phi, 0), 0).repeat(n_iterations, 0).repeat(n_components, 1)
        distribution_class = get_distribution(self._distribution_type)
        proportions, components, (m, v, s, k) = distribution_class.interpret(
            self._parameters, classes, self._sample.interval_phi)
        std = np.std(v)
        distributions = (proportions @ components)
        for i in range(n_iterations):
            copy_result = copy.copy(self)
            copy_result._distribution = distributions[i, 0]
            copy_result._proportions = proportions[i, 0]
            copy_result._components = components[i]
            copy_result._moments = (m[i], std[i], s[i], k[i])
            yield copy_result

    def _get_component(self, index: int):
        m, std, s, k = self._moments
        component = SSUResultComponent(
            self._sample.classes, self._sample.classes_phi,
            self._components[index], self._proportions[index],
            (m[index], std[index], s[index], k[index]))
        return component

    def loss(self, distance: str):
        loss_func = loss_numpy(distance)
        return loss_func(self.distribution, self.sample.distribution)

    def loss_series(self, distance: str):
        n_iterations, n_parameters, n_components = self._parameters.shape
        classes = np.expand_dims(np.expand_dims(
            self._sample.classes_phi, 0), 0).repeat(n_iterations, 0).repeat(n_components, 1)
        distribution_class = get_distribution(self._distribution_type)
        proportions, components, _ = distribution_class.interpret(
            self._parameters, classes, self._sample.interval_phi)
        distributions = (proportions @ components)[:, 0, :]
        targets = np.expand_dims(self._sample.distribution, 0).repeat(n_iterations, 0)
        loss_func = loss_numpy(distance)
        loss_series = loss_func(distributions, targets, 1)
        return loss_series
