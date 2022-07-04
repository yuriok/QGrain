import copy
import typing
from uuid import UUID, uuid4

import numpy as np
from scipy.stats import norm

from ..model import GrainSizeSample
from ._distance import get_distance_function
from ._distribution import DistributionType, get_distribution
from ._task import SSUTask


class SSUViewModel:
    def __init__(self, classes_φ, target,
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


class SSUComponentResult:
    def __init__(self, distribution: np.ndarray,
                 proportion: float,
                 moments: typing.Tuple[float, float, float, float],
                 component_args: np.ndarray):
        assert proportion is not None and np.isreal(proportion)
        # iteration may pass invalid proportion value
        # assert proportion >= 0.0 and proportion <= 1.0
        self.update(distribution, proportion, moments, component_args)

    def update(self, distribution: np.ndarray,
               proportion: float,
               moments: typing.Tuple[float, float, float, float],
               component_args: np.ndarray):
        self.__distribution = distribution
        self.__proportion = proportion
        m, v, s, k = moments
        self.__moments = dict(mean=m, std=np.sqrt(v), skewness=s, kurtosis=k)
        self.__component_args = component_args
        values_to_check = [self.__proportion]
        values_to_check.extend(self.__distribution)
        keys = ["mean", "std", "skewness", "kurtosis"]
        for key in keys:
            values_to_check.append(self.__moments[key])
        values_to_check = np.array(values_to_check)
        # if any value is nan of inf, this result is invalid
        if np.any(np.isnan(values_to_check) | np.isinf(values_to_check)):
            self.__is_valid = False
        else:
            self.__is_valid = True

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
    def component_args(self) -> np.ndarray:
        return self.__component_args

    @property
    def is_valid(self) -> bool:
        return self.__is_valid


class SSUResult:
    """
    The class to represent the fitting result of each sample.
    """
    def __init__(self, task: SSUTask,
                 func_args: typing.Iterable[float],
                 history: typing.List[np.ndarray] = None,
                 time_spent = None):
        # add uuid to manage data
        self.__uuid = uuid4()
        self.__task = task
        self.__distribution_type = task.distribution_type
        self.__n_components = task.n_components
        self.__sample = task.sample
        self.__func_args = func_args
        self.__history = [func_args] if history is None else history
        self.__components = [] # type: typing.List[SSUComponentResult]
        self.__time_spent = time_spent
        self.update(func_args)

    def update(self, func_args: typing.Iterable[float]):
        self.__func_args = func_args
        distribution_class = get_distribution(self.distribution_type)
        classes = np.expand_dims(np.expand_dims(self.sample.classes_φ, 0), 0).repeat(1, 0).repeat(self.n_components, 1)
        if np.any(np.isnan(func_args)):
            self.__distribution = np.full_like(self.sample.distribution, fill_value=np.nan)
            self.__is_valid = False
        else:
            proportions, components, (m, v, s, k) = distribution_class.interpret(func_args, classes, self.__sample.interval_φ)
            proportions, components, (m, v, s, k) = proportions[0], components[0], (m[0], v[0], s[0], k[0])
            distribution = (proportions @ components)[0]
            self.__distribution = distribution
            if len(self.__components) == 0:
                for i in range(self.n_components):
                    component_result = SSUComponentResult(components[i], proportions[0][i], (m[i], v[i], s[i], k[i]), func_args[0, :, i])
                    self.__components.append(component_result)
            else:
                for i in range(self.n_components):
                    self.__components[i].update(components[i], proportions[0][i], (m[i], v[i], s[i], k[i]), func_args[0, :, i])
            # sort by mean φ values
            # reverse is necessary
            self.__components.sort(key=lambda component: component.moments["mean"], reverse=True)

            self.__is_valid = True
            if np.any(np.isnan(self.__distribution) | np.isinf(self.__distribution)):
                self.__is_valid = False
            for component in self.__components:
                if not component.is_valid:
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
    def func_args(self) -> np.ndarray:
        return self.__func_args

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
    def components(self) -> typing.List[SSUComponentResult]:
        return self.__components

    @property
    def is_valid(self) -> bool:
        return self.__is_valid

    @property
    def history(self):
        copy_result = copy.deepcopy(self)
        for func_args in self.__history:
            copy_result.update(func_args)
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
