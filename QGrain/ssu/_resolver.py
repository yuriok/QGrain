import logging
import multiprocessing
import time
import traceback
import typing

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from scipy.optimize import OptimizeResult, basinhopping, minimize

from ..models import Dataset, Sample
from ._distance import get_distance_function
from ._distribution import DistributionType, get_distribution
from ._result import SSUResult
from ._task import SSUTask

# "cosine" has problem
built_in_distances = (
    "1-norm", "2-norm", "3-norm", "4-norm",
    "MSE", "log10MSE", "angular")

def check_distance(distance):
    assert isinstance(distance, str)
    in_set = False
    for d in built_in_distances:
        if distance == d:
            in_set = True
            break
    assert in_set


built_in_minimizers = (
    "Nelder-Mead", "Powell", "CG", "BFGS",
    "L-BFGS-B", "TNC", "SLSQP")

def check_minimizer(minimizer):
    assert isinstance(minimizer, str)
    in_set = False
    for m in built_in_minimizers:
        if minimizer == m:
            in_set = True
            break
    assert in_set


class SSUAlgorithmSetting:
    def __init__(
            self, distance: str = "log10MSE",
            minimizer: str = "SLSQP",
            try_GO: bool = False,
            GO_max_niter: int = 100,
            GO_success_niter: int = 5,  # GO - Global Optimization
            GO_step: float = 0.1,
            GO_minimizer_max_niter: int = 500,
            LO_max_niter: int = 1000):
        # validation
        check_distance(distance)
        check_minimizer(minimizer)
        assert isinstance(try_GO, bool)
        assert isinstance(GO_max_niter, int)
        assert isinstance(GO_success_niter, int)
        assert isinstance(GO_step, (int, float))
        assert isinstance(GO_minimizer_max_niter, int)
        assert isinstance(LO_max_niter, int)
        assert GO_max_niter > 0
        assert GO_success_niter > 0
        assert GO_step > 0.0
        assert GO_minimizer_max_niter > 0
        assert LO_max_niter > 0

        self.distance = distance
        self.minimizer = minimizer
        self.try_GO = try_GO
        self.GO_max_niter = GO_max_niter
        self.GO_success_niter = GO_success_niter
        self.GO_step = GO_step
        self.GO_minimizer_max_niter = GO_minimizer_max_niter
        self.LO_max_niter = LO_max_niter

    def __str__(self) -> str:
        return self.__dict__.__str__()


class BasicResolver:
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

    def local_iteration_callback(self, parameters: typing.Iterable[float]):
        pass

    def global_iteration_callback(self, parameters: typing.Iterable[float], function_value: float, accept: bool):
        pass

    def on_fitting_succeeded(self, algorithm_result: OptimizeResult, fitting_result: SSUResult):
        pass

    def get_weights(self, classes_φ: np.ndarray, distribution: np.ndarray):
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

    def try_fit(self, task: SSUTask) -> typing.Tuple[str, typing.Union[SSUTask, SSUResult]] :
        history = []
        if task.resolver_setting is None:
            setting = SSUAlgorithmSetting()
        else:
            assert isinstance(task.resolver_setting, SSUAlgorithmSetting)
            setting = task.resolver_setting
        classes = np.expand_dims(np.expand_dims(task.sample.classes_phi, 0), 0).repeat(task.n_components, 1)
        distribution_class = get_distribution(task.distribution_type)
        distance_func = get_distance_function(setting.distance)

        start_time = time.time()
        self.on_fitting_started()

        def closure(x):
            x = x.reshape((1, distribution_class.N_PARAMETERS+1, task.n_components))
            proportions, components, (m, v, s, k) = distribution_class.interpret(x, classes, task.sample.interval_phi)
            pred_distribution = (proportions[0] @ components[0]).squeeze()
            distance = distance_func(pred_distribution, task.sample.distribution)
            return distance

        def local_callback(x, *addtional):
            x = x.reshape((1, distribution_class.N_PARAMETERS+1, task.n_components))
            history.append(x)
            self.local_iteration_callback(x)

        initial_parameters = task.initial_parameters
        if task.initial_parameters is None:
            initial_parameters = np.expand_dims(distribution_class.get_defaults(task.n_components), 0)

        GO_options = {"maxiter": setting.GO_minimizer_max_niter, "disp": False}
        LO_options = {"maxiter": setting.LO_max_niter, "disp": False}

        if setting.try_GO:
            GO_minimizer_kwargs = \
                dict(method=setting.minimizer,
                     callback=local_callback,
                     options=GO_options)
            GO_result = \
                basinhopping(closure, x0=initial_parameters,
                            minimizer_kwargs=GO_minimizer_kwargs,
                            callback=self.global_iteration_callback,
                            niter_success=setting.GO_success_niter,
                            niter=setting.GO_max_niter,
                            stepsize=setting.GO_step)

            if GO_result.lowest_optimization_result.success or \
                    GO_result.lowest_optimization_result.status == 9:
                self.on_global_fitting_succeeded(GO_result)
                initial_parameters = GO_result.x
            else:
                self.on_global_fitting_failed(GO_result)
                self.on_fitting_finished()
                return GO_result.message, task

        LO_result = minimize(
            closure,
            method=setting.minimizer,
            x0=initial_parameters,
            callback=local_callback,
            options=LO_options) # type: OptimizeResult
        # Judge if the final fitting succeed
        if LO_result.success or LO_result.status == 9:
            finish_time = time.time()
            self.on_fitting_finished()
            time_spent = finish_time - start_time
            parameters = np.reshape(LO_result.x, (1, distribution_class.N_PARAMETERS+1, task.n_components))
            fitting_result = SSUResult(task, parameters, history=history, time_spent=time_spent)
            self.on_fitting_succeeded(LO_result, fitting_result)
            return LO_result.message, fitting_result
        else:
            self.on_final_fitting_failed(LO_result)
            self.on_fitting_finished()
            return LO_result.message, task


class BackgroundWorker(QObject):
    task_succeeded = Signal(SSUResult)
    task_failed = Signal(str, SSUTask)
    logger = logging.getLogger("QGrain")

    def __init__(self):
        super().__init__()

    def on_task_started(self, task: SSUTask):
        resolver = BasicResolver()
        try:
            message, result = resolver.try_fit(task)
            if isinstance(result, SSUResult):
                self.task_succeeded.emit(result)
            else:
                self.task_failed.emit(message, task)
        except Exception as e:
            self.logger.exception("An unknown exception was raised. Please check the logs for more details.", stack_info=True)
            self.task_failed.emit("An unknown exception was raised. Please check the logs for more details.", task)

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


def try_sample(
        sample: Sample,
        distribution_type: DistributionType,
        n_components: int,
        resolver_setting: SSUAlgorithmSetting = None,
        initial_parameters=None):
    task = SSUTask(sample, distribution_type, n_components, resolver_setting, initial_parameters)
    resolver = BasicResolver()
    message, result = resolver.try_fit(task)
    return result


def try_dataset(
        dataset: Dataset,
        distribution_type: DistributionType,
        n_components: int,
        resolver_setting: SSUAlgorithmSetting = None,
        initial_parameters: np.ndarray = None,
        n_processes: int = 1):
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(n_processes)
    tasks = [SSUTask(sample, distribution_type, n_components, resolver_setting, initial_parameters) for sample in dataset.samples]
    def run_task(task: SSUTask):
        resolver = BasicResolver()
        message, result = resolver.try_fit(task)
        return result
    results = pool.map(run_task, tasks)
    suceeded_results = [] # type: list[SSUResult]
    failed_tasks = [] # type: list[SSUTask]
    for result, task in zip(results, tasks):
        if isinstance(result, SSUResult):
            suceeded_results.append(suceeded_results)
        else:
            failed_tasks.append(task)
    return suceeded_results, failed_tasks
