__all__ = ["built_in_losses", "built_in_optimizers", "try_ssu", "try_dataset"]

import logging
import multiprocessing
import time
from typing import *

import numpy as np
from numpy import ndarray
from scipy.optimize import basinhopping, minimize

from .distributions import DistributionType, get_distribution, get_sorted_indexes
from .metrics import loss_numpy
from .models import Dataset, Sample, SSUResult

# "cosine" metric has problem
built_in_losses = (
    "1-norm", "2-norm", "3-norm", "4-norm",
    "mae", "mse", "rmse", "rmlse", "angular")


def check_loss(loss: str):
    assert isinstance(loss, str)
    assert loss in built_in_losses


built_in_optimizers = (
    "Nelder-Mead", "Powell", "CG", "BFGS",
    "L-BFGS-B", "TNC", "SLSQP")


def check_optimizer(optimizer: str):
    assert isinstance(optimizer, str)
    assert optimizer in built_in_optimizers


def try_ssu(sample: Sample, distribution_type: DistributionType, n_components: int, x0: ndarray = None,
            loss: str = "rmlse", optimizer: str = "SLSQP", try_global: bool = False, global_max_niter: int = 100,
            global_niter_success: int = 5, global_step_size: float = 0.2, optimizer_max_niter: int = 1000,
            need_history: bool = True, logger: logging.Logger = None,
            progress_callback: Callable[[float], None] = None) -> Tuple[Optional[SSUResult], str]:
    assert isinstance(sample, Sample)
    assert isinstance(distribution_type, DistributionType)
    assert isinstance(n_components, int)
    distribution_class = get_distribution(distribution_type)
    if x0 is None:
        x0 = np.expand_dims(distribution_class.get_defaults(n_components), 0)
    else:
        x0 = np.array(x0)
        assert x0.ndim == 2
        assert x0.shape == (distribution_class.N_PARAMETERS + 1, n_components)
    check_loss(loss)
    loss_func = loss_numpy(loss)
    check_optimizer(optimizer)
    assert isinstance(global_max_niter, int)
    assert isinstance(global_niter_success, int)
    assert isinstance(global_step_size, (int, float))
    assert isinstance(optimizer_max_niter, int)
    assert global_max_niter > 0
    assert global_niter_success > 0
    assert global_step_size > 0.0
    assert optimizer_max_niter > 0
    if logger is None:
        logger = logging.getLogger("QGrain")
    else:
        assert isinstance(logger, logging.Logger)

    start_text = f"""Performing the SSU algorithm on the sample ({sample.name}).
    Distribution type: {distribution_type.name}
    Number of components: {n_components}
    x0: {x0}
    Loss: {loss}
    Optimizer: {optimizer}
    Try global optimization: {try_global}
    Global maximum number of iterations: {global_max_niter}
    Global number of success iterations: {global_niter_success}
    Global step size: {global_step_size}
    Optimizer maximum number of iterations: {optimizer_max_niter}
    Need history: {need_history}"""
    logger.debug(start_text)

    start_time = time.time()
    global_iteration = 0
    iteration = 0
    max_iterations = global_max_niter * optimizer_max_niter if try_global else optimizer_max_niter
    history = []
    classes = np.expand_dims(np.expand_dims(sample.classes_phi, 0), 0).repeat(n_components, 1)

    def closure(x):
        x = x.reshape((1, distribution_class.N_PARAMETERS + 1, n_components))
        proportions, components, (m, v, s, k) = distribution_class.interpret(x, classes, sample.interval_phi)
        pred_distribution = (proportions[0] @ components[0]).squeeze()
        return loss_func(pred_distribution, sample.distribution)

    def callback(x: ndarray):
        nonlocal iteration
        iteration += 1
        x = x.reshape((1, distribution_class.N_PARAMETERS + 1, n_components))
        progress = iteration / max_iterations
        if need_history:
            history.append(x)
        if progress_callback is not None:
            progress_callback(progress)

    def global_callback(x: ndarray, f: float, accept: bool):
        nonlocal global_iteration
        global_iteration += 1
        x = x.reshape((1, distribution_class.N_PARAMETERS + 1, n_components))
        logger.debug(f"The global epoch {global_iteration} finished, x: {x}, function value: {f}, accepted: {accept}.")

    with np.errstate(all="ignore"):
        if try_global:
            global_result = basinhopping(
                closure, x0=x0, minimizer_kwargs=dict(
                    method=optimizer, callback=callback, options=dict(maxiter=optimizer_max_niter)),
                niter_success=global_niter_success, niter=global_max_niter,
                stepsize=global_step_size, callback=global_callback)
            if global_result.lowest_optimization_result.success or global_result.lowest_optimization_result.status == 9:
                parameters = np.reshape(global_result.x, (1, distribution_class.N_PARAMETERS + 1, n_components))
                message = global_result.message
            else:
                logger.error(f"The fitting process terminated with a error: {global_result.message}.")
                return None, global_result.message
        else:
            local_result = minimize(closure, x0=x0, method=optimizer,
                                    callback=callback, options=dict(maxiter=optimizer_max_niter))
            if local_result.success or local_result.status == 9:
                parameters = np.reshape(local_result.x, (1, distribution_class.N_PARAMETERS + 1, n_components))
                message = local_result.message
            else:
                logger.error(f"The fitting process terminated with a error: {local_result.message}.")
                return None, local_result.message

    time_spent = time.time() - start_time
    # sort the components by their grain sizes (from fine to coarse)
    sorted_indexes = get_sorted_indexes(distribution_type, parameters, classes, sample.interval_phi)
    if need_history:
        parameters = np.concatenate(history, axis=0)
    sorted_parameters = np.zeros_like(parameters)
    for i, j in enumerate(sorted_indexes):
        sorted_parameters[:, :, i] = parameters[:, :, j]
    ssu_result = SSUResult(sample, distribution_type, x0, sorted_parameters, time_spent)
    logger.debug(f"The fitting process successfully finished. {message}")
    return ssu_result, message


def try_dataset(
        dataset: Dataset,
        distribution_type: DistributionType,
        n_components: int,
        n_processes: int = 1,
        options: Dict[str, Any] = None):
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(n_processes)

    def execute(sample: Sample):
        return try_ssu(sample, distribution_type, n_components, **options)

    results = pool.map(execute, iter(dataset))
    succeeded_results: List[SSUResult] = []
    failed_samples: List[Tuple[int, str]] = []
    for i, (result, message) in enumerate(results):
        if isinstance(result, SSUResult):
            succeeded_results.append(result)
        else:
            failed_samples.append((i, message))
    return succeeded_results, failed_samples
