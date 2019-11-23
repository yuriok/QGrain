import logging
import time
from enum import Enum, unique

import numpy as np
from PySide2.QtCore import QObject, Signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from algorithms import (get_mixed_weibull, process_params, weibull, normal, normal_mean,
                        weibull_kurtosis, weibull_mean, weibull_median,
                        weibull_mode, weibull_skewness, weibull_std_deviation,
                        weibull_variance, DistributionType, get_mixed_normal)
from data import FittedData


class Resolver(QObject):
    sigSingleIterationFinished = Signal(FittedData)
    sigEpochFinished = Signal(FittedData)
    logger = logging.getLogger(name="root.Resolver")

    X_OFFSET = 0.2

    def __init__(self, distribution_type=DistributionType.Weibull, ncomp=2, auto_fit=True,
                 inherit_params=True, emit_iteration=False, time_interval=0.02,
                 display_details=False, ftol=1e-10, maxiter=100):
        super().__init__()
        # use `on_target_data_changed` to modify these values
        self.sample_name = None
        self.x = None
        self.y = None
        self.start_index = None
        self.end_index = None
        self.x_to_fit = None
        self.y_to_fit = None
        # use `on_type_changed`
        self.distribution_type = distribution_type
        # use `on_ncomp_changed`
        self.ncomp = ncomp
        self.refresh_by_settings()

        # settings
        self.auto_fit = True
        self.inherit_params = inherit_params
        self.emit_iteration = emit_iteration
        self.time_interval = time_interval
        self.display_details = display_details
        self.ftol = ftol
        self.maxiter = maxiter
        self.max_retry_time = 5
        self.max_mse = 4e-7


    # generate some necessary data for fitting
    # if `ncomp` or `distribution_type` changed, this method must be called
    def refresh_by_settings(self):
        if self.distribution_type == DistributionType.Weibull:
            (self.mixed_func, self.bounds, self.constrains,
             self.defaults, self.params) = get_mixed_weibull(self.ncomp)
            self.single_func = weibull
            self.mean_func = weibull_mean
            self.last_fitted_params = self.defaults
            self.process_fitted_params = lambda fitted_params: process_params(
                self.ncomp, self.params, fitted_params, self.distribution_type)
        else:
            (self.mixed_func, self.bounds, self.constrains,
             self.defaults, self.params) = get_mixed_normal(self.ncomp)
            self.single_func = normal
            self.mean_func = normal_mean
            self.last_fitted_params = self.defaults
            self.process_fitted_params = lambda fitted_params: process_params(
                self.ncomp, self.params, fitted_params, self.distribution_type)


    def on_ncomp_changed(self, ncomp: int):
        self.ncomp = ncomp
        self.refresh_by_settings()
        self.logger.debug("Component Number has been changed to [%d].", ncomp)

    def on_type_changed(self, distribution_type: DistributionType):
        self.distribution_type = distribution_type
        self.refresh_by_settings()
        self.logger.debug("Distribution type has been changed to [%s].", distribution_type)

    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            self.__setattr__(setting, value)
        self.logger.debug("Settings have been changed. [%s]", kwargs)

    def on_target_data_changed(self, sample_name, x, y):
        if x is None:
            raise ValueError(x)
        if y is None:
            raise ValueError(y)
        if type(x) != np.ndarray:
            raise TypeError(x)
        if type(y) != np.ndarray:
            raise TypeError(y)
        self.logger.debug("Target data has been changed to [%s].", sample_name)
        self.sample_name = sample_name
        self.x = x
        self.y = y
        self.start_index, self.end_index = self.get_valid_data_range()
        self.x_to_fit, self.y_to_fit = self.get_processed_data()
        if self.auto_fit:
            self.try_fit()
    
    def get_squared_sum_of_residual_errors(self, values, targets):
        errors = np.sum(np.square(values - targets))
        return errors

    def get_mean_squared_errors(self, values, targets):
        mse = np.mean(np.square(values - targets))
        return mse

    def get_valid_data_range(self):
        start_index = 0
        end_index = -1
        for i, value in enumerate(self.y):
            if value > 0.0:
                start_index = i
                break
        for i, value in enumerate(self.y[start_index+1:], start_index+1):
            if value == 0.0:
                end_index = i
                break
        self.logger.debug("The index of valid data ranges from [%d] to [%d] (exclusive).", start_index, end_index)
        return start_index, end_index

    def get_processed_data(self):
        x_to_fit = np.array(
            range(self.end_index-self.start_index)) + self.X_OFFSET
        y_to_fit = self.y[self.start_index: self.end_index] / 100
        self.logger.debug("The raw data has been processed.")
        return x_to_fit, y_to_fit

    def get_fitted_data(self, fitted_params):
        real_x = self.x[self.start_index:self.end_index]
        # the target data to fit
        target = (real_x, self.y_to_fit)
        # the fitted sum data of all components
        fitted_sum = (real_x, self.mixed_func(self.x_to_fit, *fitted_params))
        # the fitted data of each single component
        processed_params = self.process_fitted_params(fitted_params)
        components = []
        for beta, eta, fraction in processed_params:
            components.append((real_x, self.single_func(
                self.x_to_fit, beta, eta)*fraction))

        # get the relationship (func) to convert x_to_fit to real x
        x_to_real = interp1d(self.x_to_fit, real_x)
        statistic = []
        
        for i, (beta, eta, fraction) in enumerate(processed_params):
            try:
                # use max operation to convert np.ndarray to float64
                mean_value = x_to_real(self.mean_func(beta, eta)).max()
                median_value = x_to_real(weibull_median(beta, eta)).max()
                mode_value = x_to_real(weibull_mode(beta, eta)).max()
            except ValueError:
                mean_value = np.nan
                median_value = np.nan
                mode_value = np.nan
            statistic.append({
                "name": "C{0}".format(i+1),
                "beta": beta,
                "eta": eta,
                "loc": self.start_index,
                "x_offset": self.X_OFFSET,
                "fraction": fraction,
                "mean": mean_value,
                "median": median_value,
                "mode": mode_value,
                "variance": weibull_variance(beta, eta),
                "standard_deviation": weibull_std_deviation(beta, eta),
                "skewness": weibull_skewness(beta, eta),
                "kurtosis": weibull_kurtosis(beta, eta)
            })

        mse = np.mean(np.square(target[1] - fitted_sum[1]))
        fitted_data = FittedData(self.sample_name, target, fitted_sum, mse, components, statistic)
        # self.logger.debug("One shot of fitting has finished, current mean squared error [%E].", mse)
        return fitted_data

    def iteration_callback(self, fitted_params):
        if self.emit_iteration:
            time.sleep(self.time_interval)
            self.sigSingleIterationFinished.emit(
                self.get_fitted_data(fitted_params))

    def try_fit(self):
        def closure(args):
            current_values = self.mixed_func(self.x_to_fit, *args)
            return self.get_squared_sum_of_residual_errors(current_values, self.y_to_fit)*100

        params_to_fit = self.defaults
        if self.inherit_params:
            params_to_fit = self.last_fitted_params
        self.logger.debug("Fitting task started, max mean squared error is limited to [%E], and max retry time is [%d].", self.max_mse, self.max_retry_time)
        ftol = self.ftol
        for i in range(self.max_retry_time):
            # "L-BFGS-B", "SLSQP", "TNC"
            fitted_result = minimize(closure, method="SLSQP", x0=params_to_fit,
                                    bounds=self.bounds, constraints=self.constrains, callback=self.iteration_callback,
                                    options={"maxiter": self.maxiter, "disp": self.display_details, "ftol": ftol})
            # update this variable to inherit the fitted params
            params_to_fit = fitted_result.x
            mse = self.get_mean_squared_errors(self.mixed_func(self.x_to_fit, *fitted_result.x), self.y_to_fit)
            self.logger.debug("Try %d: mean squared error is %E.", i, mse)
            if mse < self.max_mse:
                break
            else:
                ftol = ftol*1e-50
            
        self.last_fitted_params = params_to_fit
        self.logger.info("The epoch of fitting has finished, the fitted parameters are: [%s]", fitted_result.x)
        self.sigEpochFinished.emit(self.get_fitted_data(fitted_result.x))
