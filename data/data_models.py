from __future__ import annotations

import uuid
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import kendalltau, pearsonr, spearmanr

from algorithms import AlgorithmData, DistributionType


class SampleData:
    def __init__(self, name, distribution: np.ndarray):
        self.name = name
        self.distribution = distribution


class GrainSizeData:
    def __init__(self, is_valid=False, classes: np.ndarray = None,
                 sample_data_list: Iterable[SampleData] = None):
        self.is_valid = is_valid
        self.classes = classes
        self.sample_data_list = sample_data_list


class ComponentFittingResult:
    def __init__(self, fitting_space_x: np.ndarray, params: Tuple[float], fraction: float,
                 algorithm_data: AlgorithmData, x_to_real: Callable):
        self.params = params
        self.fraction = fraction
        self.component_y = algorithm_data.single_func(fitting_space_x, *params) * fraction
        try:
            temp_value = algorithm_data.mean(*params)
            self.mean = x_to_real(temp_value).max()
            self.median = x_to_real(algorithm_data.median(*params)).max()
            self.mode = x_to_real(algorithm_data.mode(*params)).max()
        except ValueError:
            self.mean = np.nan
            self.median = np.nan
            self.mode = np.nan
        self.variance = algorithm_data.variance(*params)
        self.standard_deviation = algorithm_data.standard_deviation(*params)
        self.skewness = algorithm_data.skewness(*params)
        self.kurtosis = algorithm_data.kurtosis(*params)

    @property
    def has_nan(self) -> bool:
        values_to_check = [*self.params, self.fraction,
                           self.mean, self.median,
                           self.mode, self.variance,
                           self.standard_deviation, self.skewness, self.kurtosis]
        if np.any(np.isnan(self.component_y)) or np.any(np.isnan(values_to_check)):
            return True
        else:
            return False


class FittingResult:
    def __init__(self, name: str, real_x: np.ndarray,
                 fitting_space_x: np.ndarray, bin_numbers: np.ndarray,
                 target_y: np.ndarray, algorithm_data: AlgorithmData,
                 fitted_params: Iterable, x_offset: float):
        length = len(real_x)
        assert len(fitting_space_x) == length
        assert len(bin_numbers) == length
        assert len(target_y) == length

        # some identifications
        # add uuid to manage data
        self.uuid = uuid.uuid4()
        self.name = name
        self.distribution_type = algorithm_data.distribution_type
        self.component_number = algorithm_data.component_number
        self.param_count = algorithm_data.get_param_count()
        self.param_names = algorithm_data.get_param_names()
        # orgianl data
        self.real_x = real_x
        self.fitting_space_x = fitting_space_x
        self.bin_numbers = bin_numbers
        self.target_y = target_y

        # fitted data
        self.fitted_y = algorithm_data.mixed_func(fitting_space_x-x_offset, *fitted_params)
        self.components = [] # type: List[FittedComponentData]
        processed_params = algorithm_data.process_params(fitted_params, x_offset)
        x_to_real = interp1d(fitting_space_x, self.real_x)
        for params, fraction in processed_params:
            self.components.append(ComponentFittingResult(fitting_space_x, params, fraction, algorithm_data, x_to_real))
        # sort by mean values
        self.components.sort(key=lambda component: component.mean)

        # some test for fitting result
        self.error_array = self.target_y-self.fitted_y
        self.mean_squared_error = np.mean(np.square(self.error_array))
        # https://scipy.github.io/devdocs/generated/scipy.stats.pearsonr.html
        self.pearson_r = pearsonr(self.target_y, self.fitted_y)
        # https://scipy.github.io/devdocs/generated/scipy.stats.kendalltau.html
        self.kendall_tau = kendalltau(self.target_y, self.fitted_y)
        # https://scipy.github.io/devdocs/generated/scipy.stats.spearmanr.html
        self.spearman_r = spearmanr(self.target_y, self.fitted_y)

    def has_invalid_value(self) -> bool:
        if np.any(np.isnan(self.fitted_y)):
            return True
        for component in self.components:
            if component.has_nan:
                return True
        return False
