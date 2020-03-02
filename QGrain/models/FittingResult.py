__all__ = ["ComponentFittingResult", "FittingResult"]

import copy
from typing import Callable, Dict, Iterable, List, Tuple
from uuid import UUID, uuid4

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import kendalltau, pearsonr, spearmanr

from QGrain.algorithms import AlgorithmData, DistributionType


class ComponentFittingResult:
    """
    The class to record the fitting result of each component.

    All its attributes have been packed with read-only properties to avoid the modification by mistake.

    Use `ctor` to initialize the instance and treat it as immutable object.
    If you have sufficient reasons to modify its attributes, please use the `update` method of it.

    Attributes:
        real_x: The `numpy.ndarray` which represents the real (raw) values of x (i.e. grain size classes).
        fitting_space_x: The `numpy.ndarray` which represents the x values (e.g. bin numbers) used in fitting process.
        params: The paramters which were used to calculate the y array (i.e. distribution) and statistic values of each component.
        fraction: The fraction of this component.
        mean, median, mode, etc: The statistic values of this component.
    """
    def __init__(self, real_x: np.ndarray, fitting_space_x: np.ndarray,
                 algorithm_data: AlgorithmData,
                 params: Iterable[float], fraction: float):
        assert np.isreal(fraction) and fraction is not None
        # iteration may pass invalid fraction value
        # assert fraction >= 0.0 and fraction <= 1.0

        self.__real_x = real_x
        self.__fitting_space_x = fitting_space_x
        if np.any(np.isnan(params)):
            self.__fraction = np.nan
            self.__component_y = np.full_like(fitting_space_x, fill_value=np.nan)
            self.__mean = np.nan
            self.__median = np.nan
            self.__mode = np.nan
            self.__variance = np.nan
            self.__standard_deviation = np.nan
            self.__skewness = np.nan
            self.__kurtosis = np.nan
        else:
            self.update(algorithm_data, params, fraction)

    def update(self, algorithm_data: AlgorithmData, params: Iterable[float], fraction: float):
        self.__params = params
        self.__fraction = fraction
        self.__component_y = algorithm_data.single_func(self.__fitting_space_x, *params) * fraction
        x_to_real = interp1d(self.__fitting_space_x, self.__real_x)
        try:
            self.__mean = x_to_real(algorithm_data.mean(*params)).max()
        except ValueError:
            self.__mean = np.nan
        try:
            self.__median = x_to_real(algorithm_data.median(*params)).max()
        except ValueError:
            self.__median = np.nan
        try:
            self.__mode = x_to_real(algorithm_data.mode(*params)).max()
        except ValueError:
            self.__mode = np.nan
        self.__variance = algorithm_data.variance(*params)
        self.__standard_deviation = algorithm_data.standard_deviation(*params)
        self.__skewness = algorithm_data.skewness(*params)
        self.__kurtosis = algorithm_data.kurtosis(*params)

    @property
    def params(self) -> Tuple[float]:
        return self.__params

    @property
    def fraction(self) -> float:
        return self.__fraction

    @property
    def component_y(self) -> np.ndarray:
        return self.__component_y

    @property
    def mean(self) -> float:
        return self.__mean

    @property
    def median(self) -> float:
        return self.__median

    @property
    def mode(self) -> float:
        return self.__mode

    @property
    def variance(self) -> float:
        return self.__variance

    @property
    def standard_deviation(self) -> float:
        return self.__standard_deviation

    @property
    def skewness(self) -> float:
        return self.__skewness

    @property
    def kurtosis(self) -> float:
        return self.__kurtosis

    @property
    def has_nan(self) -> bool:
        values_to_check = [self.fraction, self.mean,
                           self.median, self.mode,
                           self.variance, self.standard_deviation,
                           self.skewness, self.kurtosis]
        if np.any(np.isnan(self.component_y)) or np.any(np.isnan(values_to_check)):
            return True
        else:
            return False


class FittingResult:
    """
    The class to represent the fitting result of each sample.
    """
    def __init__(self, name: str, real_x: np.ndarray,
                 fitting_space_x: np.ndarray, bin_numbers: np.ndarray,
                 target_y: np.ndarray, algorithm_data: AlgorithmData,
                 fitted_params: Iterable[float], x_offset: float,
                 fitting_history: List[np.ndarray] = None):
        length = len(real_x)
        assert len(fitting_space_x) == length
        assert len(bin_numbers) == length
        assert len(target_y) == length

        # add uuid to manage data
        self.__uuid = uuid4()
        self.__name = name
        self.__real_x = real_x
        self.__fitting_space_x = fitting_space_x
        self.__bin_numbers = bin_numbers
        self.__x_offset = x_offset
        self.__target_y = target_y
        self.__distribution_type = algorithm_data.distribution_type
        self.__component_number = algorithm_data.component_number
        self.__param_count = algorithm_data.param_count
        self.__param_names = algorithm_data.param_names
        self.__fitting_history = [fitted_params] if fitting_history is None else fitting_history
        self.__components = [] # type: List[ComponentFittingResult]
        self.update(fitted_params)


    def update(self, fitted_params: Iterable[float]):
        algorithm_data = AlgorithmData.get_algorithm_data(self.distribution_type, self.component_number)
        if np.any(np.isnan(fitted_params)):
            self.__fitted_y = np.full_like(self.__fitting_space_x, fill_value=np.nan)
            self.__error_array = np.full_like(self.__fitting_space_x, fill_value=np.nan)
            self.__mean_squared_error = np.nan
            self.__pearson_r = (np.nan, np.nan)
            self.__kendall_tau = (np.nan, np.nan)
            self.__spearman_r = (np.nan, np.nan)
        else:
            self.__fitted_y = algorithm_data.mixed_func(self.__fitting_space_x - self.__x_offset, *fitted_params)
            # some test for fitting result
            self.__error_array = self.__target_y - self.__fitted_y
            self.__mean_squared_error = np.mean(np.square(self.__error_array))
            # https://scipy.github.io/devdocs/generated/scipy.stats.pearsonr.html
            self.__pearson_r = pearsonr(self.__target_y, self.__fitted_y)
            # https://scipy.github.io/devdocs/generated/scipy.stats.kendalltau.html
            self.__kendall_tau = kendalltau(self.__target_y, self.__fitted_y)
            # https://scipy.github.io/devdocs/generated/scipy.stats.spearmanr.html
            self.__spearman_r = spearmanr(self.__target_y, self.__fitted_y)

        processed_params = algorithm_data.process_params(fitted_params, self.__x_offset)
        if  len(self.__components) == 0:
            for params, fraction in processed_params:
                component_result = ComponentFittingResult(self.real_x, self.fitting_space_x, algorithm_data, params, fraction)
                self.__components.append(component_result)
        else:
            for component, (params, fraction) in zip(self.__components, processed_params):
                component.update(algorithm_data, params, fraction)
        # sort by mean values
        self.__components.sort(key=lambda component: component.mean)

    @property
    def uuid(self) -> UUID:
        return self.__uuid

    @property
    def name(self) -> str:
        return self.__name

    @property
    def real_x(self) -> np.ndarray:
        return self.__real_x

    @property
    def fitting_space_x(self) -> np.ndarray:
        return self.__fitting_space_x

    @property
    def bin_numbers(self) -> np.ndarray:
        return self.__bin_numbers

    @property
    def target_y(self) -> np.ndarray:
        return self.__target_y

    @property
    def distribution_type(self) -> DistributionType:
        return self.__distribution_type

    @property
    def component_number(self) -> int:
        return self.__component_number

    @property
    def param_count(self) -> int:
        return self.__param_count

    @property
    def param_names(self) -> Tuple[str]:
        return self.__param_names

    @property
    def components(self) -> Iterable[ComponentFittingResult]:
        return [component for component in self.__components]

    @property
    def fitted_y(self) -> np.ndarray:
        return self.__fitted_y

    @property
    def mean_squared_error(self) -> float:
        return self.__mean_squared_error

    @property
    def pearson_r(self) -> Tuple[float, float]:
        return self.__pearson_r

    @property
    def kendall_tau(self) -> Tuple[float, float]:
        return self.__kendall_tau

    @property
    def spearman_r(self) -> Tuple[float, float]:
        return self.__spearman_r

    @property
    def has_invalid_value(self) -> bool:
        if np.any(np.isnan(self.__fitted_y)):
            return True
        for component in self.__components:
            if component.has_nan:
                return True
        return False

    @property
    def history(self):
        copy_result = copy.deepcopy(self)
        for fitted_params in self.__fitting_history:
            copy_result.update(fitted_params)
            yield copy_result

    @property
    def iteration_number(self):
        return len(self.__fitting_history)
