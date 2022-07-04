import copy
import typing
from uuid import UUID, uuid4

import numpy as np

from ..model import GrainSizeSample
from ._distance import get_distance_function
from ._distribution import DistributionType, get_distribution
from ._task import SSUTask


class SSUViewModel:
    def __init__(
            self, classes_φ, target,
            mixed, distributions, proportions,
            component_prefix="C", title="", **kwargs):
        self.classes_φ = classes_φ
        self.mixed = mixed
        self.distributions = distributions
        self.proportions = proportions
        self.target=target
        self.component_prefix = component_prefix
        self.title = title
        self.kwargs = kwargs

    @property
    def n_components(self) -> int:
        return len(self.distributions)


class SSUResultComponent:
    def __init__(
            self, distribution: np.ndarray,
            proportion: float,
            moments: typing.Tuple[float, float, float, float],
            parameters: np.ndarray):
        self.update(distribution, proportion, moments, parameters)

    def update(
            self, distribution: np.ndarray,
            proportion: float,
            moments: typing.Tuple[float, float, float, float],
            parameters: np.ndarray):
        self.__distribution = distribution
        self.__proportion = proportion
        m, v, s, k = moments
        self.__moments = dict(mean=m, std=np.sqrt(v), skewness=s, kurtosis=k)
        self.__parameters = parameters

    @property
    def proportion(self) -> float:
        return self.__proportion

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution

    @property
    def moments(self) -> dict:
        return self.__moments

    @property
    def parameters(self) -> np.ndarray:
        return self.__parameters


class SSUResult:
    """
    This class represents the SSU result of each sample.
    """

    def __init__(
            self, task: SSUTask,
            parameters: typing.Iterable[float],
            history: typing.List[np.ndarray] = None,
            time_spent=None):
        # add uuid to manage data
        self.__uuid = uuid4()
        self.__task = task
        self.__distribution_type = task.distribution_type
        self.__n_components = task.n_components
        self.__sample = task.sample
        self.__parameters = parameters
        self.__history = [parameters] if history is None else history
        self.__components = [] # type: typing.List[SSUResultComponent]
        self.__time_spent = time_spent
        self.update(parameters)

    def update(self, parameters: typing.Iterable[float]):
        self.__parameters = parameters
        classes = np.expand_dims(np.expand_dims(self.sample.classes_φ, 0), 0).repeat(1, 0).repeat(self.n_components, 1)
        if np.any(np.isnan(parameters)):
            self.__distribution = None
            self.__is_valid = False
        else:
            proportions, components, (m, v, s, k) = get_distribution(self.distribution_type).interpret(parameters, classes, self.__sample.interval_φ)
            proportions, components, (m, v, s, k) = proportions[0], components[0], (m[0], v[0], s[0], k[0])
            distribution = (proportions @ components)[0]
            self.__distribution = distribution
            if len(self.__components) == 0:
                for i in range(self.n_components):
                    component_result = SSUResultComponent(components[i], proportions[0][i], (m[i], v[i], s[i], k[i]), parameters[0, :, i])
                    self.__components.append(component_result)
            else:
                for i in range(self.n_components):
                    self.__components[i].update(components[i], proportions[0][i], (m[i], v[i], s[i], k[i]), parameters[0, :, i])
            # sort by mean φ values
            # reverse is necessary
            self.__components.sort(key=lambda component: component.moments["mean"], reverse=True)

            self.__is_valid = True
            for values in [proportions, components, distribution, m, v, s, k]:
                if np.any(np.logical_or(np.isnan(values), np.isinf(values))):
                    self.__is_valid = False
                    break

    @property
    def uuid(self) -> UUID:
        return self.__uuid

    @property
    def sample(self) -> GrainSizeSample:
        return self.__sample

    @property
    def classes_μm(self) -> np.ndarray:
        return self.__sample.classes_μm

    @property
    def classes_φ(self) -> np.ndarray:
        return self.__sample.classes_φ

    @property
    def task(self) -> SSUTask:
        return self.__task

    @property
    def parameters(self) -> np.ndarray:
        return self.__parameters

    @property
    def distribution_type(self) -> DistributionType:
        return self.__distribution_type

    @property
    def n_components(self) -> int:
        return self.__n_components

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution

    @property
    def components(self) -> typing.List[SSUResultComponent]:
        return self.__components

    @property
    def is_valid(self) -> bool:
        return self.__is_valid

    @property
    def history(self):
        copy_result = copy.deepcopy(self)
        for parameters in self.__history:
            copy_result.update(parameters)
            yield copy_result

    def get_distance(self, distance: str):
        distance_func = get_distance_function(distance)
        distance = distance_func(self.distribution, self.sample.distribution)
        return distance

    def get_distance_series(self, distance: str):
        distance_series = []
        for result in self.history:
            distance_series.append(result.get_distance(distance))
        return distance_series

    @property
    def time_spent(self):
        return self.__time_spent

    @property
    def n_iterations(self):
        return len(self.__history)

    @property
    def view_model(self) -> SSUViewModel:
        distributions = [component.distribution for component in self.components]
        proportions = [comp.proportion for comp in self.components]
        return SSUViewModel(
            self.sample.classes_φ, self.sample.distribution,
            self.distribution, distributions, proportions,
            component_prefix="C", title=self.sample.name)

    @property
    def view_models(self) -> typing.Iterable[SSUViewModel]:
        for result in self.history:
            yield result.view_model
