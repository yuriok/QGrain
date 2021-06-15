import copy
import time
import traceback
import typing
from uuid import UUID, uuid4

import numpy as np
import torch
from PySide2.QtCore import QObject, QThread, Signal, Slot
from scipy.optimize import OptimizeResult, basinhopping, minimize
from scipy.stats import norm
from torch.nn import Module, Parameter, ReLU, Softmax

from QGrain import DistributionType, FittingState
from QGrain.distributions import BaseDistribution
from QGrain.distributions import get_distance_func_by_name as distance_func_numpy
from QGrain.kernels import KERNEL_MAP, N_PARAMS_MAP
from QGrain.kernels import get_distance_func_by_name as distance_func_torch
from QGrain.kernels import get_initial_guess
from QGrain.models.ClassicResolverSetting import ClassicResolverSetting
from QGrain.models.GrainSizeSample import GrainSizeSample
from QGrain.models.NNResolverSetting import NNResolverSetting
from QGrain.statistic import geometric, logarithmic

INVALID_STATISTIC = dict(mean=np.nan, std=np.nan, skewness=np.nan, kurtosis=np.nan)

class SSUTask:
    def __init__(self, sample: GrainSizeSample,
                 distribution_type: DistributionType,
                 n_components: int,
                 resolver="classic",
                 resolver_setting=None,
                 initial_guess=None,
                 reference=None):
        self.uuid = uuid4()
        self.sample = sample
        self.distribution_type = distribution_type
        self.n_components = n_components
        self.resolver = resolver
        self.resolver_setting = resolver_setting
        self.initial_guess = initial_guess
        self.reference = reference

class SSUViewModel:
    def __init__(self, classes_φ, target,
                 mixed, distributions, fractions,
                 component_prefix="C", title="", **kwargs):
        self.classes_φ = classes_φ
        self.mixed = mixed
        self.distributions = distributions
        self.fractions = fractions
        self.target=target
        self.component_prefix = component_prefix
        self.title = title
        self.kwargs = kwargs

    @property
    def n_components(self) -> int:
        return len(self.distributions)

class ComponentResult:
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
            self.__geometric_moments = INVALID_STATISTIC
            self.__logarithmic_moments = INVALID_STATISTIC
            self.__is_valid = False
        else:
            self.__fraction = fraction
            self.__distribution = distribution.single_function(sample.classes_φ, *func_args)
            self.__geometric_moments = geometric(sample.classes_μm, self.__distribution)
            self.__logarithmic_moments= logarithmic(sample.classes_φ, self.__distribution)

            values_to_check = [self.__fraction]
            values_to_check.extend(self.__distribution)
            keys = ["mean", "std", "skewness", "kurtosis"]
            for key in keys:
                values_to_check.append(self.__geometric_moments[key])
                values_to_check.append(self.__logarithmic_moments[key])
            values_to_check = np.array(values_to_check)
            # if any value is nan of inf, this result is invalid
            if np.any(np.isnan(values_to_check) | np.isinf(values_to_check)):
                self.__is_valid = False
            else:
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

def get_demo_view_model() -> SSUViewModel:
    classes_μm = np.logspace(0, 5, 101)*0.02
    classes_φ = -np.log2(classes_μm/1000.0)
    mixed = np.zeros_like(classes_φ)
    locs = [10.5, 7.5, 5.0]
    scales = [1.0, 1.0, 1.0]
    fractions = [0.2, 0.3, 0.5]
    distributions = []
    interval = abs((classes_φ[-1] - classes_φ[0]) / (len(classes_φ) - 1))
    for loc, scale, fraction in zip(locs, scales, fractions):
        distribution = norm.pdf(classes_φ, loc=loc, scale=scale) * interval
        distributions.append(distribution)
        mixed += distribution * fraction
    model = SSUViewModel(classes_φ, mixed,
                                        mixed, distributions, fractions,
                                        title="Demo")
    return model

class SSUResult:
    """
    The class to represent the fitting result of each sample.
    """
    def __init__(self, task: SSUTask,
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
        self.__components = [] # type: typing.List[ComponentResult]
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
                    component_result = ComponentResult(self.__sample, distribution, func_args, fraction)
                    self.__components.append(component_result)
            else:
                for component, (func_args, fraction) in zip(self.__components, unpacked_args):
                    component.update(self.__sample, distribution, func_args, fraction)
            # sort by mean φ values
            # reverse is necessary
            self.__components.sort(key=lambda component: component.logarithmic_moments["mean"], reverse=True)

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
    def distribution_type(self) -> DistributionType:
        return self.__distribution_type

    @property
    def n_components(self) -> int:
        return self.__n_components

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution

    @property
    def components(self) -> typing.List[ComponentResult]:
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
        distance_func = distance_func_numpy(distance)
        distribution = BaseDistribution.get_distribution(self.__distribution_type, self.__n_components)
        values = distribution.mixed_function(self.__sample.classes_φ, *self.__mixed_func_args)
        targets = self.__sample.distribution
        distance = distance_func(values, targets)
        return distance

    def get_distance_series(self, distance: str):
        distance_func = distance_func_numpy(distance)
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
    def view_model(self) -> SSUViewModel:
        distributions = [comp.distribution for comp in self.components]
        fractions = [comp.fraction for comp in self.components]
        return SSUViewModel(
            self.sample.classes_φ, self.sample.distribution,
            self.distribution, distributions, fractions,
            component_prefix="C", title=self.sample.name)

    @property
    def view_models(self) -> typing.Iterable[SSUViewModel]:
        for result in self.history:
            yield result.view_model

class NNMixedDistribution(Module):
    def __init__(self, distribution_type: DistributionType, n_components: int, initial_guess=None):
        super().__init__()
        self.distribution_type = distribution_type
        self.n_components = n_components
        self.softmax = Softmax(dim=0)
        self.relu = ReLU()
        self.kernels = []
        n_params = N_PARAMS_MAP[distribution_type]
        kernel_class = KERNEL_MAP[distribution_type]
        if initial_guess is None:
            self.fractions = Parameter(torch.rand(n_components), requires_grad=True)
            for i in range(n_components):
                kernel = kernel_class()
                self.__setattr__(f"kernel_{i+1}", kernel)
                self.kernels.append(kernel)
        else:
            assert len(initial_guess) == (n_params+1) * n_components - 1
            expanded = list(initial_guess)
            expanded.append(1-sum(expanded[-n_components+1:]))
            self.fractions = Parameter(torch.Tensor(expanded[-n_components:]), requires_grad=True)
            for i in range(n_components):
                kernel = kernel_class(*expanded[i*n_params:(i+1)*n_params])
                self.__setattr__(f"kernel_{i+1}", kernel)
                self.kernels.append(kernel)

    @property
    def params(self):
        params = []
        for kernel in self.kernels:
            params.extend(kernel.params)
        with torch.no_grad():
            params.extend([value.item() for value in self.softmax(self.fractions)[:-1]])
            return params

    def forward(self, classes_φ):
        fractions = self.softmax(self.fractions).reshape(1, -1)
        distributions = []
        for i in range(self.n_components):
            distribution = self.kernels[i](classes_φ)
            distributions.append(distribution.reshape(1, -1))
        distributions = torch.cat(distributions, dim=0)
        mixed = fractions @ distributions
        return mixed, fractions, distributions

class NNResolver:
    def __init__(self):
        pass

    def try_fit(self, task: SSUTask):
        assert task.resolver == "neural"
        if task.resolver_setting is None:
            setting = NNResolverSetting()
        else:
            assert isinstance(task.resolver_setting, NNResolverSetting)
            setting = task.resolver_setting

        initial_guess = task.initial_guess
        if task.reference is not None:
            initial_guess = get_initial_guess(task.distribution_type, task.reference)

        start = time.time()
        X = torch.from_numpy(np.array(task.sample.classes_φ, dtype=np.float32))
        y = torch.from_numpy(np.array(task.sample.distribution, dtype=np.float32))
        X, y = X.to(setting.device), y.to(setting.device)
        model = NNMixedDistribution(task.distribution_type, task.n_components, initial_guess=initial_guess).to(setting.device)
        distance = distance_func_torch(setting.distance)
        optimizer = torch.optim.Adam(model.parameters(), lr=setting.lr, eps=setting.eps)

        def train():
            # Compute prediction error
            y_hat, _, _ = model(X)
            loss = distance(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()
        epoch = 0
        loss_series = []
        history = []
        while True:
            loss_value = train()
            loss_series.append(loss_value)
            history.append(model.params)
            epoch += 1
            if epoch < setting.min_niter:
                continue
            if loss_value < setting.tol:
                break
            delta_loss = np.std(loss_series[-50:])
            if delta_loss < setting.ftol:
                break
            if epoch > setting.max_niter:
                break
        time_spent = time.time() - start
        finished_params = model.params
        history.append(finished_params)
        result = SSUResult(task, finished_params, history=history, time_spent=time_spent)
        return FittingState.Succeeded, result

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
        distance = distance_func_numpy(setting.distance)
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

class BackgroundWorker(QObject):
    task_succeeded = Signal(SSUResult)
    task_failed = Signal(str, SSUTask)

    def __init__(self):
        super().__init__()

    def on_task_started(self, task: SSUTask):
        if task.resolver == "classic":
            resolver = ClassicResolver()
        elif task.resolver == "neural":
            resolver = NNResolver()
        else:
            raise NotImplementedError(task.resolver)
        try:
            state, result = resolver.try_fit(task)
            if state == FittingState.Succeeded:
                self.task_succeeded.emit(result)
            else:
                self.task_failed.emit(f"Fitting Failed, error details:\n{result.__str__()}", task)
        except Exception as e:
            self.task_failed.emit(f"Unknown Exception Raised: {type(e)}, {e.__str__()}, {traceback.format_exc()}", task)

class AsyncWorker(QObject):
    task_started = Signal(SSUTask)

    def __init__(self):
        super().__init__()
        self.background_worker = BackgroundWorker()
        self.working_thread = QThread()
        self.background_worker.moveToThread(self.working_thread)
        self.task_started.connect(self.background_worker.on_task_started)
        self.background_worker.task_failed.connect(self.on_task_failed)
        self.background_worker.task_succeeded.connect(self.on_task_succeeded)
        self.working_thread.start()

    def on_task_succeeded(self, fitting_result: SSUResult):
        pass

    def on_task_failed(self, failed_info, task):
        pass

    @Slot()
    def execute_task(self, task: SSUTask):
        self.task_started.emit(task)
