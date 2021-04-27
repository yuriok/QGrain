__all__ = ["ComponentFittingResult", "FittingResult"]

import copy
import typing
from uuid import UUID, uuid4

import numpy as np
from QGrain.algorithms import DistributionType
from QGrain.algorithms.distributions import BaseDistribution, get_distance_func_by_name
from QGrain.algorithms.moments import get_moments, invalid_moments
from QGrain.models.FittingTask import FittingTask
from QGrain.models.GrainSizeSample import GrainSizeSample
from QGrain.models.MixedDistributionChartViewModel import MixedDistributionChartViewModel

class ComponentFittingResult:
    def __init__(self, sample: GrainSizeSample,
                 distribution: BaseDistribution,
                 func_args: typing.Iterable[float], fraction: float):
        assert fraction is not None and np.isreal(fraction)
        # iteration may pass invalid fraction value
        # assert fraction >= 0.0 and fraction <= 1.0
        self.update(sample, distribution, func_args, fraction)

    def update(self, sample: GrainSizeSample, distribution: BaseDistribution, func_args: typing.Iterable[float], fraction: float):
        if np.any(np.isnan(func_args)):
            self.__fraction = np.nan
            self.__distribution = np.full_like(sample.distribution, fill_value=np.nan)
            self.__geometric_moments = invalid_moments
            self.__logarithmic_moments = invalid_moments
            self.__is_valid = False
        else:
            self.__fraction = fraction
            self.__distribution = distribution.single_function(sample.classes_φ, *func_args)
            self.__geometric_moments, self.__logarithmic_moments= get_moments(sample.classes_μm, sample.classes_φ, self.__distribution, FW57=False)
            self.__is_valid = True

    @property
    def fraction(self) -> float:
        return self.__fraction

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution

    @property
    def geometric_moments(self) -> dict:
        return self.__geometric_moments

    @property
    def logarithmic_moments(self) -> dict:
        return self.__logarithmic_moments

    @property
    def is_valid(self) -> bool:
        return self.__is_valid

class FittingResult:
    """
    The class to represent the fitting result of each sample.
    """
    def __init__(self, task: FittingTask,
                 mixed_func_args: typing.Iterable[float],
                 history: typing.List[np.ndarray] = None,
                 time_spent = None):
        # add uuid to manage data
        self.__uuid = uuid4()
        self.__task = task
        self.__distribution_type = task.distribution_type
        self.__n_components = task.n_components
        self.__sample = task.sample
        self.__mixed_func_args = mixed_func_args
        self.__history = [mixed_func_args] if history is None else history
        self.__components = [] # type: List[ComponentFittingResult]
        self.__time_spent = time_spent
        self.update(mixed_func_args)

    def update(self, mixed_func_args: typing.Iterable[float]):
        distribution = BaseDistribution.get_distribution(self.__distribution_type, self.__n_components)

        if np.any(np.isnan(mixed_func_args)):
            self.__distribution = np.full_like(self.__sample.distribution, fill_value=np.nan)
            self.__is_valid = False
        else:
            self.__distribution = distribution.mixed_function(self.__sample.classes_φ, *mixed_func_args)
            unpacked_args = distribution.unpack_parameters(mixed_func_args)
            if len(self.__components) == 0:
                for func_args, fraction in unpacked_args:
                    component_result = ComponentFittingResult(self.__sample, distribution, func_args, fraction)
                    self.__components.append(component_result)
            else:
                for component, (func_args, fraction) in zip(self.__components, unpacked_args):
                    component.update(self.__sample, distribution, func_args, fraction)
            # sort by mean φ values
            # reverse is necessary
            self.__components.sort(key=lambda component: component.logarithmic_moments["mean"], reverse=True)
            self.__is_valid = True

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
    def task(self) -> FittingTask:
        return self.__task

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
    def components(self) -> typing.List[ComponentFittingResult]:
        return self.__components

    @property
    def is_valid(self) -> bool:
        return self.__is_valid

    @property
    def mixed_func_args(self) -> np.ndarray:
        return self.__mixed_func_args

    @property
    def history(self):
        copy_result = copy.deepcopy(self)
        for fitted_params in self.__history:
            copy_result.update(fitted_params)
            yield copy_result

    def get_distance(self, distance: str):
        distance_func = get_distance_func_by_name(distance)
        distribution = BaseDistribution.get_distribution(self.__distribution_type, self.__n_components)
        values = distribution.mixed_function(self.__sample.classes_φ, *self.__mixed_func_args)
        targets = self.__sample.distribution
        distance = distance_func(values, targets)
        return distance

    def get_distance_series(self, distance: str):
        distance_func = get_distance_func_by_name(distance)
        distribution = BaseDistribution.get_distribution(self.__distribution_type, self.__n_components)
        distance_series = []
        for func_args in self.__history:
            values = distribution.mixed_function(self.__sample.classes_φ, *func_args)
            targets = self.__sample.distribution
            distance = distance_func(values, targets)
            distance_series.append(distance)
        return distance_series

    @property
    def last_func_args(self):
        return self.__history[-1]

    @property
    def time_spent(self):
        return self.__time_spent

    @property
    def n_iterations(self):
        return len(self.__history)

    @property
    def view_model(self) -> MixedDistributionChartViewModel:
        distributions = [comp.distribution for comp in self.components]
        fractions = [comp.fraction for comp in self.components]
        return MixedDistributionChartViewModel(
            self.sample.classes_φ, self.sample.distribution,
            self.distribution, distributions, fractions,
            component_prefix="C", title=self.sample.name)

    @property
    def view_models(self) -> typing.Iterable[MixedDistributionChartViewModel]:
        for result in self.history:
            yield result.view_model
