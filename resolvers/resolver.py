from enum import Enum, unique

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping, minimize

from algorithms import DistributionType, MixedDistributionData
from data import FittedData


@unique
class DataValidationResult(Enum):
    Valid = -1
    NameNone = 0
    NameEmpty = 1
    XNone = 2
    YNone = 3
    XTypeInvalid = 4
    YTypeInvalid = 5
    XHasNan = 6
    YHasNan = 7
    LengthNotEqual = 8


class Resolver:
    def __init__(self, global_optimization_maxiter=100,
                 global_optimization_success_iter=3,
                 global_optimization_stepsize = 1,
                 final_tolerance=1e-100,
                 final_maxiter=1000,
                 minimizer_tolerance=1e-8,
                 minimizer_maxiter=500):
        self.__distribution_type = DistributionType.GeneralWeibull
        self.__component_number = 3
        self.__algorithm_data_cache = {}
        self.refresh()

        self.global_optimization_maxiter = global_optimization_maxiter
        self.global_optimization_success_iter = global_optimization_success_iter
        self.global_optimization_stepsize = global_optimization_stepsize

        self.minimizer_tolerance = minimizer_tolerance
        self.minimizer_maxiter = minimizer_maxiter

        self.final_tolerance = final_tolerance
        self.final_maxiter = final_maxiter

        self.sample_name = None
        self.real_x = None
        self.x_offset = 0
        self.fitting_space_x = None
        self.target_y = None

        self.start_index = None
        self.end_index = None

        self.x_to_fit = None
        self.y_to_fit = None

    @property
    def distribution_type(self):
        return self.__distribution_type

    @distribution_type.setter
    def distribution_type(self, value: DistributionType):
        if type(value) != DistributionType:
            return
        self.__distribution_type = value
        self.refresh()

    @property
    def component_number(self):
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
        key = (self.distribution_type, self.component_number)
        if key in self.__algorithm_data_cache.keys():
            self.algorithm_data = self.__algorithm_data_cache[key]
        else:
            algorithm_data = AlgorithmData(*key)
            self.__algorithm_data_cache.update({key: algorithm_data})
            self.algorithm_data = algorithm_data
        self.initial_guess = self.algorithm_data.defaults

    @staticmethod
    def get_squared_sum_of_residual_errors(values, targets):
        errors = np.sum(np.square(values - targets))
        return errors

    @staticmethod
    def get_mean_squared_errors(values, targets):
        mse = np.mean(np.square(values - targets))
        return mse

    @staticmethod
    def get_valid_data_range(target_y, slice_data=True):
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
            # search for tail to head
            for i, value in enumerate(target_y[start_index+1:][::-1]):
                if value > 0.0:
                    if i <= 1:
                    break
                    else:
                        end_index = (i-1)*(-1)
                        break
        return start_index, end_index

    @staticmethod
    def validate_data(sample_name: str, x: np.ndarray, y: np.ndarray) -> DataValidationResult:
        if sample_name is None:
            return DataValidationResult.NameNone
        if sample_name == "":
            return DataValidationResult.NameEmpty
        if x is None:
            return DataValidationResult.XNone
        if y is None:
            return DataValidationResult.YNone
        if type(x) != np.ndarray:
            return DataValidationResult.XTypeInvalid
        if type(y) != np.ndarray:
            return DataValidationResult.YTypeInvalid
        if len(x) != len(y):
            return DataValidationResult.LengthNotEqual
        if np.any(np.isnan(x)):
            return DataValidationResult.XHasNan
        if np.any(np.isnan(y)):
            return DataValidationResult.YHasNan

        return DataValidationResult.Valid

    # hooks
    def on_data_invalid(self, sample_name: str, x: np.ndarray, y: np.ndarray, validation_result: DataValidationResult):
        pass

    def on_data_fed(self, sample_name):
        pass

    def on_data_not_prepared(self):
        pass

    def on_fitting_started(self):
        pass

    def on_fitting_finished(self):
        pass

    def on_global_fitting_failed(self, fitted_result):
        pass

    def on_final_fitting_failed(self, fitted_result):
        pass

    def on_exception_raised_while_fitting(self, exception):
        pass

    def local_iteration_callback(self, fitted_params):
        pass

    def global_iteration_callback(self, fitted_params, function_value, accept):
        pass

    def on_fitting_succeeded(self, fitted_result):
        pass

    def preprocess_data(self):
        # Normal and General Weibull distribution need to use x offset to get better performance
        self.start_index, self.end_index = Resolver.get_valid_data_range(self.target_y)
        if self.distribution_type == DistributionType.Normal or \
                self.distribution_type == DistributionType.GeneralWeibull:
            self.x_offset = self.start_index
        else:
            self.x_offset = 0.0

        self.bin_numbers = np.array(range(len(self.target_y)), dtype=np.float64) + 1

        # fitting under the bin numbers' space
        if self.distribution_type == DistributionType.Normal or \
                self.distribution_type == DistributionType.Weibull or \
                self.distribution_type == DistributionType.GeneralWeibull:
            self.fitting_space_x = self.bin_numbers
        else:
            raise NotImplementedError(self.distribution_type)

    def feed_data(self, sample_name: str, x: np.ndarray, y: np.ndarray):
        validation_result = Resolver.validate_data(sample_name, x, y)
        if validation_result is not DataValidationResult.Valid:
            self.on_data_invalid(sample_name, x, y, validation_result)
            return
        self.sample_name = sample_name
        self.real_x = x
        self.target_y = y
        self.preprocess_data()
        self.on_data_fed(sample_name)

    def change_settings(self, **kwargs):
        for key, value in kwargs.items():
            if key == "global_optimization_maxiter":
                self.global_optimization_maxiter = value
            elif key == "global_optimization_success_iter":
                self.global_optimization_success_iter = value
            elif key == "global_optimization_stepsize":
                self.global_optimization_stepsize = value
            elif key == "minimizer_tolerance":
                self.minimizer_tolerance = value
            elif key == "minimizer_maxiter":
                self.minimizer_maxiter = value
            elif key == "final_tolerance":
                self.final_tolerance = value
            elif key == "final_maxiter":
                self.final_maxiter = value
            else:
                raise NotImplementedError(key)

    def get_fitting_result(self, fitted_params):
        result = FittingResult(self.sample_name, self.real_x,
                                 self.fitting_space_x, self.bin_numbers,
                                 self.target_y, self.algorithm_data,
                                 fitted_params, self.x_offset)
        return result

        mse = Resolver.get_mean_squared_errors(target[1], fitted_sum[1])
        # TODO: add more test for difference between observation and fitting
        fitted_data = FittedData(self.sample_name, self.distribution_type, target, fitted_sum, mse, components, statistic)
        return fitted_data

    def try_fit(self):
        if self.x_to_fit is None or self.y_to_fit is None:
            self.on_data_not_prepared()
            return
        self.on_fitting_started()

        def closure(args):
            current_values = self.mixed_data.mixed_func(self.x_to_fit, *args)
            return Resolver.get_squared_sum_of_residual_errors(current_values, self.y_to_fit)*100

        minimizer_kwargs = dict(method="SLSQP",
                                bounds=self.mixed_data.bounds, constraints=self.mixed_data.constrains,
                                callback=self.local_iteration_callback,
                                options={"maxiter": self.minimizer_maxiter, "ftol": self.minimizer_tolerance})
        try:
            global_fitted_result = basinhopping(closure, x0=self.initial_guess,
                                                minimizer_kwargs=minimizer_kwargs,
                                                callback=self.global_iteration_callback,
                                                niter_success=self.global_optimization_success_iter,
                                                niter=self.global_optimization_maxiter,
                                                stepsize=self.global_optimization_stepsize)

            # the basinhopping method do not implement the `OptimizeResult` correctly
            # it don't contains `success`
            if global_fitted_result.lowest_optimization_result.success or global_fitted_result.lowest_optimization_result.status == 9:
                pass
            else:
                self.on_global_fitting_failed(global_fitted_result)
                self.on_fitting_finished()
                return
            fitted_result = minimize(closure, method="SLSQP", x0=global_fitted_result.x,
                                     bounds=self.mixed_data.bounds, constraints=self.mixed_data.constrains,
                                     callback=self.local_iteration_callback,
                                     options={"maxiter": self.final_maxiter, "ftol": self.final_tolerance})
            # judge if the final fitting succeed
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html
            if fitted_result.success or fitted_result.status == 9:
                self.on_fitting_succeeded(fitted_result)
                self.on_fitting_finished()
                return
            else:
                self.on_final_fitting_failed(fitted_result)
                self.on_fitting_finished()
                return
        except Exception as e:
            self.on_exception_raised_while_fitting(e)
            self.on_fitting_finished()
