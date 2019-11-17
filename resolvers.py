import time
from enum import Enum, unique

import numpy as np
from PyQt5.QtCore import QMutex, QObject, pyqtSignal
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from algorithms import get_mixed_weibull, mean, median, process_params, weibull


@unique
class BaseDistribution(Enum):
    LogNormal = 1
    Weibull = 2


class Resolver(QObject):
    sigSingleIterationFinished = pyqtSignal(list)
    sigEpochFinished = pyqtSignal(list)
    
    X_OFFSET = 0.2
    def __init__(self, distribution_type=BaseDistribution.Weibull, ncomp=3, emit_iteration=False, time_interval=0.1, display_details=False, ftol=1e-100, maxiter=1000):
        super().__init__()
        self.__mutex = QMutex()
        self.__cancelFitting = False
        # use `on_target_data_changed` to modify these values
        self.__sample_id = None
        self.__x = None
        self.__y = None
        self.__start = None
        self.__end = None
        self.__x_to_fit = None
        self.__y_to_fit = None
        # use `on_type_changed`
        self.__distribution_type = distribution_type
        # use `on_ncomp_changed`
        self.__ncomp = ncomp
        self.refresh_by_settings()

        # settings
        self.emit_iteration = emit_iteration
        self.time_interval = time_interval
        self.display_details = display_details
        self.ftol = ftol
        self.maxiter = maxiter

    # generate some necessary data for fitting
    # if `ncomp` or `distribution_type` changed, this method must be called
    def refresh_by_settings(self):
        self.__mutex.lock()
        if self.__distribution_type == BaseDistribution.Weibull:
            self.mixed_func, self.bounds, self.constrains, self.defaults, self.params = get_mixed_weibull(self.__ncomp)
            self.single_func = weibull
            self.last_fitted_params = self.defaults
            self.process_fitted_params = lambda fitted_params: process_params(self.__ncomp, self.params, fitted_params)
        else:
            self.__mutex.unlock()
            raise NotImplementedError(self.__distribution_type)
        self.__mutex.unlock()


    def on_ncomp_changed(self, ncomp: int):
        self.__mutex.lock()
        self.__ncomp = ncomp
        self.__cancelFitting = True
        self.__mutex.unlock()
        self.refresh_by_settings()


    def on_type_changed(self, distribution_type: BaseDistribution):
        self.__mutex.lock()
        self.__distribution_type = distribution_type
        self.__cancelFitting = True
        self.__mutex.unlock()
        self.refresh_by_settings()


    def on_target_data_changed(self, sample_id, x, y):
        if x is None:
            raise ValueError(x)
        if y is None:
            raise ValueError(y)
        if type(x) != np.ndarray:
            raise TypeError(x)
        if type(y) != np.ndarray:
            raise TypeError(y)
        self.__mutex.lock()
        self.__x = x
        self.__y = y
        self.__start, self.__end = self.get_valid_data_range()
        self.__x_to_fit, self.__y_to_fit = self.get_processed_data()
        # when x and y changed, the fitting task should be canceled
        self.__cancelFitting = True
        self.__mutex.unlock()
        # and it's no need to call `refresh_by_settings`


    def get_squared_sum_of_residual_errors(self, values, targets):
        errors = np.sum(np.square(values - targets))
        return errors


    def get_valid_data_range(self):
        start_index = 0
        end_index = -1
        for i, value in enumerate(self.__y):
            if value > 0.0:
                start_index = i
                break
        for i, value in enumerate(self.__y[start_index+1:], start_index+1):
            if value == 0.0:
                end_index = i
                break
        return start_index, end_index


    def get_processed_data(self):
        partial_x = np.array(range(self.__end-self.__start)) + self.X_OFFSET
        partial_y = self.__y[self.__start: self.__end] / 100
        return partial_x, partial_y


    def get_visualization_data(self, fitted_params):
        # calculate data for visualization
        visualization_data = []
        visualization_x = self.__x[self.__start:self.__end]
        # the target data to fit
        visualization_data.append((visualization_x, self.__y_to_fit))
        # the fitted sum data of all components
        visualization_data.append((visualization_x, self.mixed_func(self.__x_to_fit, *fitted_params)))
        # the fitted data of each single component
        processed_params = self.process_fitted_params(fitted_params)
        for beta, eta, fraction in processed_params:
            visualization_data.append((visualization_x, self.single_func(self.__x_to_fit, beta, eta)*fraction))

        # get the relationship (func) to convert x_to_fit to real x
        x_to_real = interp1d(self.__x_to_fit, visualization_x)
        statistic_values = []
        for beta, eta, fraction in processed_params:
            statistic_values.append({
                "fraction": fraction,
                "mean": x_to_real(mean(beta, eta)),
                "median": x_to_real(median(beta, eta))})

        visualization_data.append(statistic_values)
        return visualization_data


    def iteration_callback(self, fitted_params):
        if self.__cancelFitting:
            self.__mutex.lock()
            self.__cancelFitting = False
            self.__mutex.unlock()
            return

        if self.emit_iteration:
            time.sleep(self.time_interval)
            self.sigSingleIterationFinished.emit(self.get_visualization_data(fitted_params))


    def try_fit(self):
        def closure(args):
            current_values = self.mixed_func(self.__x_to_fit, *args)
            return self.get_squared_sum_of_residual_errors(current_values, self.__y_to_fit)*100

        fitted_result = minimize(closure, method="SLSQP", x0=self.last_fitted_params, bounds=self.bounds, constraints=self.constrains,
                       callback=self.iteration_callback, options={"maxiter": self.maxiter, "disp": self.display_details, "ftol": self.ftol})
        
        # update this variable to inherit the fitted params
        self.last_fitted_params = fitted_result.x
        visualization_data = self.get_visualization_data(fitted_result.x)
        self.sigEpochFinished.emit(visualization_data)
