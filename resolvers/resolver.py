import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from algorithms import *
from data import FittedData


class Resolver:

    def __init__(self, global_optimization_maxiter=100,
                 global_optimization_success_iter=5, final_tolerance=1e-100,
                 final_maxiter=1000, minimizer_tolerance=1e-8, minimizer_maxiter=500):
        self.__distribution_type = DistributionType.Weibull
        self.__ncomp = 2
        self.refresh()

        self.global_optimization_maxiter = global_optimization_maxiter
        self.global_optimization_success_iter = global_optimization_success_iter

        self.final_tolerance = final_tolerance
        self.final_maxiter = final_maxiter

        self.minimizer_tolerance = minimizer_tolerance
        self.minimizer_maxiter = minimizer_maxiter

        self.real_x = None
        self.y_data = None

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
    def ncomp(self):
        return self.__ncomp

    @ncomp.setter
    def ncomp(self, value: int):
        if type(value) != int:
            return
        if value <= 1:
            return
        self.__ncomp = value
        self.refresh()

    # generate some necessary data for fitting
    # if `ncomp` or `distribution_type` changed, this method must be called
    def refresh(self):
        if self.distribution_type == DistributionType.Weibull:
            (self.mixed_func, self.bounds, self.constrains,
             self.defaults, self.params) = get_mixed_weibull(self.ncomp)
            self.initial_guess = self.defaults
            self.process_fitted_params = lambda fitted_params: process_params(
                self.ncomp, self.params, fitted_params, self.distribution_type)

            # TODO: it's no need when only ncomp changed
            self.single_func = weibull
            self.mean_func = weibull_mean
            self.median_func = weibull_median
            self.mode_func = weibull_mode
            self.variance_func = weibull_variance
            self.std_deviation_func = weibull_std_deviation
            self.skewness_func = weibull_skewness
            self.kurtosis_func = weibull_kurtosis

        else:
            raise NotImplementedError(self.distribution_type)

    def get_fitted_data(self, fitted_params):
        partial_real_x = self.real_x[self.start_index:self.end_index]
        # the target data to fit
        target = (partial_real_x, self.y_to_fit)
        # the fitted sum data of all components
        fitted_sum = (partial_real_x, self.mixed_func(
            self.x_to_fit, *fitted_params))
        # the fitted data of each single component
        processed_params = self.process_fitted_params(fitted_params)
        components = []
        for beta, eta, fraction in processed_params:
            components.append((partial_real_x, self.single_func(
                self.x_to_fit, beta, eta)*fraction))

        # get the relationship (func) to convert x_to_fit to real x
        x_to_real = interp1d(self.x_to_fit, partial_real_x)
        statistic = []

        # TODO: the params number of each component may vary between different distribution type
        for i, (beta, eta, fraction) in enumerate(processed_params):
            try:
                # use max operation to convert np.ndarray to float64
                mean_value = x_to_real(self.mean_func(beta, eta)).max()
                median_value = x_to_real(self.median_func(beta, eta)).max()
                mode_value = x_to_real(self.mode_func(beta, eta)).max()
            except ValueError:
                mean_value = np.nan
                median_value = np.nan
                mode_value = np.nan
            # TODO: maybe not some distribution types has not all statistic values
            statistic.append({
                "name": "C{0}".format(i+1),
                "beta": beta,
                "eta": eta,
                "x_offset": self.start_index+1,
                "fraction": fraction,
                "mean": mean_value,
                "median": median_value,
                "mode": mode_value,
                "variance": self.variance_func(beta, eta),
                "standard_deviation": self.std_deviation_func(beta, eta),
                "skewness": self.skewness_func(beta, eta),
                "kurtosis": self.kurtosis_func(beta, eta)
            })

        mse = Resolver.get_mean_squared_errors(target[1], fitted_sum[1])
        # TODO: add more test for difference between observation and fitting
        fitted_data = FittedData(self.sample_name, target, fitted_sum, mse, components, statistic)
        # self.logger.debug("One shot of fitting has finished, current mean squared error [%E].", mse)
        return fitted_data

    @staticmethod
    def get_squared_sum_of_residual_errors(values, targets):
        errors = np.sum(np.square(values - targets))
        return errors

    @staticmethod
    def get_mean_squared_errors(values, targets):
        mse = np.mean(np.square(values - targets))
        return mse

    @staticmethod
    def get_valid_data_range(y_data):
        start_index = 0
        end_index = -1
        for i, value in enumerate(y_data):
            if value > 0.0:
                start_index = i
                break
        for i, value in enumerate(y_data[start_index+1:], start_index+1):
            if value == 0.0:
                end_index = i
                break
        return start_index, end_index

    def preprocess_data(self):
        self.start_index, self.end_index = Resolver.get_valid_data_range(self.y_data)
        if self.distribution_type == DistributionType.Weibull:
            length = len(self.y_data)
            self.x_to_fit = np.array(
                range(self.end_index-self.start_index)) + 1
            self.y_to_fit = self.y_data[self.start_index: self.end_index]
        else:
            # TODO: Add support for other distributions
            raise NotImplementedError(self.distribution_type)

    @staticmethod
    def validate_data(x, y) -> bool:
        # TODO: add more validation
        if len(x) != len(y):
            return False

        return True

    def feed_data(self, x, y):
        if not Resolver.validate_data(x, y):
            # TODO: handle exceptions
            return

        self.real_x = x
        self.y_data = y
        self.preprocess_data()

    def iteration_callback(self, params, fitting_state):
        pass

    # TODO: add more hocks

    def try_fit(self):
        if self.x_to_fit is None or self.y_to_fit is None:
            return

        def closure(args):
            current_values = self.mixed_func(self.x_to_fit, *args)
            return Resolver.get_squared_sum_of_residual_errors(current_values, self.y_to_fit)*100

        minimizer_kwargs = dict(method="SLSQP",
                                bounds=self.bounds, constraints=self.constrains, callback=self.iteration_callback,
                                options={"maxiter": self.minimizer_maxiter, "ftol": self.minimizer_tolerance})

        global_fitted_result = basinhopping(closure, x0=self.initial_guess, minimizer_kwargs=minimizer_kwargs,
                                            niter_success=self.global_optimization_success_iter, niter=self.global_optimization_maxiter)

        fitted_result = minimize(closure, method="SLSQP", x0=global_fitted_result.x,
                                 bounds=self.bounds, constraints=self.constrains, callback=self.iteration_callback,
                                 options={"maxiter": self.minimizer_maxiter, "ftol": self.final_tolerance})

        return fitted_result
