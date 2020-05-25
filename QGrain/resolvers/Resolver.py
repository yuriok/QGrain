__all__ = ["Resolver"]

from enum import Enum, unique
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, basinhopping, minimize

from QGrain.algorithms import AlgorithmData, DistributionType
from QGrain.models.AlgorithmSettings import AlgorithmSettings
from QGrain.models.FittingResult import FittingResult
from QGrain.models.SampleData import SampleData


class Resolver:
    """
    The base class of resolvers.
    """
    def __init__(self):
        self.__distribution_type = DistributionType.GeneralWeibull
        self.__component_number = 3
        self.refresh()

        # algorithms settings
        self.algorthm_settings = AlgorithmSettings()

        # the related data of current sample
        self.sample_name = None # type: str
        self.real_x = None # type: np.ndarray
        self.x_offset = 0.0 # type: float
        self.bin_numbers = None # type: np.ndarray
        self.fitting_space_x = None # type: np.ndarray
        self.target_y = None # type: np.ndarray
        # parameters to preprocess the data
        self.start_index = None # type: int
        self.end_index = None # type: int
        self.fitting_history = None # type: List[np.ndarray]

    @property
    def distribution_type(self) -> DistributionType:
        return self.__distribution_type

    @distribution_type.setter
    def distribution_type(self, value: DistributionType):
        if type(value) != DistributionType:
            return
        self.__distribution_type = value
        self.refresh()

    @property
    def component_number(self) -> int:
        return self.__component_number

    @component_number.setter
    def component_number(self, value: int):
        if type(value) != int:
            return
        if value < 1:
            return
        self.__component_number = value
        self.refresh()

    def refresh(self):
        self.algorithm_data = AlgorithmData.get_algorithm_data(self.distribution_type, self.component_number)
        self.initial_guess = self.algorithm_data.defaults

    @staticmethod
    def get_squared_sum_of_residual_errors(
            values: np.ndarray, targets: np.ndarray) -> float:
        errors = np.sum(np.square(values - targets))
        return errors

    @staticmethod
    def get_valid_data_range(target_y: np.ndarray, slice_data: bool=True):
        start_index = 0
        end_index = len(target_y)
        if slice_data:
            for i, value in enumerate(target_y):
                if value > 0.0:
                    if i == 0:
                        break
                    else:
                        start_index = i-1
                        break
            # search from tail to head
            for i, value in enumerate(target_y[start_index+1:][::-1]):
                if value > 0.0:
                    if i <= 1:
                        break
                    else:
                        end_index = (i-1)*(-1)
                        break
        return start_index, end_index

    # hooks
    def on_data_fed(self, sample_name: str):
        pass

    def on_data_not_prepared(self):
        pass

    def on_fitting_started(self):
        self.fitting_history = []

    def on_fitting_finished(self):
        pass

    def on_global_fitting_failed(self, algorithm_result: OptimizeResult):
        pass

    def on_global_fitting_succeeded(self, algorithm_result: OptimizeResult):
        pass

    def on_final_fitting_failed(self, algorithm_result: OptimizeResult):
        pass

    def on_exception_raised_while_fitting(self, exception: Exception):
        pass

    def local_iteration_callback(self, fitted_params: Iterable[float]):
        self.fitting_history.append(fitted_params)

    def global_iteration_callback(self, fitted_params: Iterable[float], function_value: float, accept: bool):
        pass

    def on_fitting_succeeded(self, algorithm_result: OptimizeResult):
        pass

    def preprocess_data(self):
        self.start_index, self.end_index = Resolver.get_valid_data_range(self.target_y)
        # Normal and General Weibull distribution need to use x offset to get better performance.
        # Because if do not do that, the search space will be larger,
        # and hence increse the difficulty of searching.
        if self.distribution_type == DistributionType.Normal or \
                self.distribution_type == DistributionType.GeneralWeibull:
            self.x_offset = self.start_index
        else:
            self.x_offset = 0.0
        # Bin numbers are similar to log(x), but will not raise negative value.
        # This will make the fitting of some distributions (e.g. Weibull) easier.
        self.bin_numbers = np.array(range(len(self.target_y)), dtype=np.float64) + 1
        # fitting under the bin numbers' space
        if self.distribution_type == DistributionType.Normal or \
                self.distribution_type == DistributionType.Weibull or \
                self.distribution_type == DistributionType.GeneralWeibull:
            self.fitting_space_x = self.bin_numbers
        else:
            raise NotImplementedError(self.distribution_type)

    def feed_data(self, sample: SampleData):
        self.sample_name = sample.name
        self.real_x = sample.classes
        self.target_y = sample.distribution
        self.preprocess_data()
        self.on_data_fed(sample.name)

    @property
    def data_prepared(self) -> bool:
        if self.real_x is None:
            # all these attributes should be `None`
            # otherwise the codes are incorrect
            assert self.sample_name is None
            assert self.target_y is None
            assert self.fitting_space_x is None
            assert self.bin_numbers is None
            assert self.start_index is None
            assert self.end_index is None
            return False
        else:
            return True

    def change_settings(self, settings: AlgorithmSettings):
        assert settings is not None
        assert isinstance(settings, AlgorithmSettings)
        self.algorthm_settings = settings

    def get_fitting_result(self, fitted_params: Iterable[float],
                           fitting_history: List[np.ndarray] = None):
        result = FittingResult(self.sample_name, self.real_x,
                               self.fitting_space_x, self.bin_numbers,
                               self.target_y, self.algorithm_data,
                               fitted_params, self.x_offset,
                               fitting_history=self.fitting_history if fitting_history is None else fitting_history)
        return result

    def try_fit(self):
        if not self.data_prepared:
            self.on_data_not_prepared()
            return
        self.on_fitting_started()

        def closure(args):
            # using partial values (i.e. don't use unnecessary zero values)
            # will highly improve the performance of algorithms
            x_to_fit = self.fitting_space_x[self.start_index: self.end_index]-self.x_offset
            y_to_fit = self.target_y[self.start_index: self.end_index]
            current_values = self.algorithm_data.mixed_func(x_to_fit, *args)
            return Resolver.get_squared_sum_of_residual_errors(current_values, y_to_fit)*100

        global_optimization_minimizer_kwargs = \
            dict(method="SLSQP",
                 bounds=self.algorithm_data.bounds,
                 constraints=self.algorithm_data.constrains,
                 callback=self.local_iteration_callback,
                 options={"maxiter": self.algorthm_settings.global_optimization_minimizer_maximum_iteration,
                          "ftol": 10**-self.algorthm_settings.global_optimization_minimizer_tolerance_level})
        try:
            global_optimization_result = \
                basinhopping(closure, x0=self.initial_guess,
                             minimizer_kwargs=global_optimization_minimizer_kwargs,
                             callback=self.global_iteration_callback,
                             niter_success=self.algorthm_settings.global_optimization_success_iteration,
                             niter=self.algorthm_settings.global_optimization_maximum_iteration,
                             stepsize=self.algorthm_settings.global_optimization_step_size)

            if global_optimization_result.lowest_optimization_result.success or \
                    global_optimization_result.lowest_optimization_result.status == 9:
                self.on_global_fitting_succeeded(global_optimization_result)
            else:
                self.on_global_fitting_failed(global_optimization_result)
                self.on_fitting_finished()
                return
            final_optimization_result = \
                minimize(closure, method="SLSQP",
                         x0=global_optimization_result.x,
                         bounds=self.algorithm_data.bounds,
                         constraints=self.algorithm_data.constrains,
                         callback=self.local_iteration_callback,
                         options={"maxiter": self.algorthm_settings.final_optimization_minimizer_maximum_iteration,
                                  "ftol": 10**-self.algorthm_settings.final_optimization_minimizer_tolerance_level})
            # judge if the final fitting succeed
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html
            if final_optimization_result.success or final_optimization_result.status == 9:
                self.on_fitting_succeeded(final_optimization_result)
                self.on_fitting_finished()
                return
            else:
                self.on_final_fitting_failed(final_optimization_result)
                self.on_fitting_finished()
                return
        except Exception as e:
            self.on_exception_raised_while_fitting(e)
            self.on_fitting_finished()
