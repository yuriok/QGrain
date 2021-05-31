__all__ = ["ClassicResolver"]

import time
import typing

import numpy as np
from QGrain.algorithms import FittingState
from QGrain.distributions import BaseDistribution, get_distance_func_by_name
from QGrain.models.ClassicResolverSetting import ClassicResolverSetting
from QGrain.ssu import SSUResult, SSUTask
from scipy.optimize import OptimizeResult, basinhopping, minimize


class ClassicResolver:

    def __init__(self, hooks=None):
        if hooks is not None:
            for name, func in hooks.items():
                self.__setattr__(name, func)

    # hooks
    def on_fitting_started(self):
        pass

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

    def local_iteration_callback(self, fitted_params: typing.Iterable[float]):
        pass

    def global_iteration_callback(self, fitted_params: typing.Iterable[float], function_value: float, accept: bool):
        pass

    def on_fitting_succeeded(self, algorithm_result: OptimizeResult, fitting_result: SSUResult):
        pass

    def get_weights(self, classes_φ, distribution):
        from scipy.signal import find_peaks
        peaks, info = find_peaks(distribution)
        non_zeros = np.argwhere(distribution)
        start, end = non_zeros[0].max(), non_zeros[-1].max()
        weights = np.ones_like(distribution)
        weights[start: start + 3] += 2.0
        weights[end - 3: end] += 2.0
        for peak in peaks:
            weights[peak - 2: peak + 2] += 2.0
        return weights

    def try_fit(self, task: SSUTask) -> typing.Tuple[FittingState, object]:
        assert task.resolver == "classic"
        history = []
        distribution = BaseDistribution.get_distribution(task.distribution_type, task.n_components)
        if task.resolver_setting is None:
            setting = ClassicResolverSetting()
        else:
            assert isinstance(task.resolver_setting, ClassicResolverSetting)
            setting = task.resolver_setting
        distance = get_distance_func_by_name(setting.distance)
        start_time = time.time()
        self.on_fitting_started()
        use_weights = False
        if use_weights:
            weights = self.get_weights(task.sample.classes_φ, task.sample.distribution)
            def closure(params):
                params[-task.n_components:] = np.abs(params[-task.n_components:])
                current_values = distribution.mixed_function(task.sample.classes_φ, *params)
                return distance(current_values*weights, task.sample.distribution*weights)
        else:
            def closure(params):
                params[-task.n_components:] = np.abs(params[-task.n_components:])
                current_values = distribution.mixed_function(task.sample.classes_φ, *params)
                return distance(current_values, task.sample.distribution)

        def local_callback(mixed_func_args, *addtional):
            history.append(mixed_func_args)
            self.local_iteration_callback(mixed_func_args)

        initial_guess = task.initial_guess
        if task.initial_guess is None:
            initial_guess = np.array(distribution.defaults)

        if task.reference is not None:
            assert len(task.reference) == task.n_components
            initial_guess = BaseDistribution.get_initial_guess(task.distribution_type, task.reference)

        if setting.minimizer == "trust-constr":
            GO_options = {"maxiter": setting.GO_minimizer_max_niter,
                    #    "ftol": setting.GO_minimizer_ftol,
                        "disp": False}
            FLO_options = {"maxiter": setting.FLO_max_niter,
                    #    "ftol": setting.FLO_ftol,
                        "disp": False}
        else:
            GO_options = {"maxiter": setting.GO_minimizer_max_niter,
                        "ftol": setting.GO_minimizer_ftol,
                        "disp": False}
            FLO_options = {"maxiter": setting.FLO_max_niter,
                        "ftol": setting.FLO_ftol,
                        "disp": False}

        if setting.try_GO:
            global_optimization_minimizer_kwargs = \
                dict(method=setting.minimizer,
                     tol=setting.GO_minimizer_tol,
                     bounds=distribution.bounds,
                     constraints=distribution.constrains,
                     callback=local_callback,
                     options=GO_options)

            GO_result = \
                basinhopping(closure, x0=initial_guess,
                            minimizer_kwargs=global_optimization_minimizer_kwargs,
                            callback=self.global_iteration_callback,
                            niter_success=setting.GO_success_niter,
                            niter=setting.GO_max_niter,
                            stepsize=setting.GO_step)

            if GO_result.lowest_optimization_result.success or \
                    GO_result.lowest_optimization_result.status == 9:
                self.on_global_fitting_succeeded(GO_result)
                initial_guess = GO_result.x
            else:
                self.on_global_fitting_failed(GO_result)
                self.on_fitting_finished()
                return FittingState.Failed, GO_result

        FLO_result = \
            minimize(closure, method=setting.minimizer,
                    x0=initial_guess,
                    tol=setting.FLO_tol,
                    bounds=distribution.bounds,
                    constraints=distribution.constrains,
                    callback=local_callback,
                    options=FLO_options)
        # judge if the final fitting succeed
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html
        # When the minimizer is "Nelder-Mead", it will return failed result if it has reached the max niter
        if FLO_result.success or FLO_result.status == 9 or setting.minimizer == "Nelder-Mead" or setting.minimizer == "trust-constr":
            finish_time = time.time()
            self.on_fitting_finished()
            time_spent = finish_time - start_time
            fitting_result = SSUResult(task, FLO_result.x, history=history, time_spent=time_spent)
            self.on_fitting_succeeded(FLO_result, fitting_result)
            return FittingState.Succeeded, fitting_result
        else:
            self.on_final_fitting_failed(FLO_result)
            self.on_fitting_finished()
            return FittingState.Failed, FLO_result
