__all__ = ["built_in_losses", "built_in_optimizers", "try_ssu", "try_dataset"]

import logging
import multiprocessing
import time
from typing import *

import numpy as np
from numpy import ndarray
from scipy.stats import wasserstein_distance
from scipy.optimize import basinhopping, minimize

from .models import DistributionType, Dataset, Sample, SSUResult, ArtificialSample, ArtificialDataset
from .distributions import get_distribution, sort_components
from .metrics import loss_numpy

# "cosine" metric has problem
built_in_losses = (
    "1-norm", "2-norm", "3-norm", "4-norm",
    "mae", "mse", "rmse", "rmlse", "lmse",
    "angular", "cosine", "wasserstein")


def check_loss(loss: str):
    assert isinstance(loss, str)
    assert loss in built_in_losses


built_in_optimizers = (
    "Nelder-Mead", "Powell", "CG", "BFGS",
    "L-BFGS-B", "TNC", "SLSQP")


def check_optimizer(optimizer: str):
    assert isinstance(optimizer, str)
    assert optimizer in built_in_optimizers


def try_ssu(sample: Union[ArtificialSample, Sample], distribution_type: DistributionType, n_components: int,
            x0: ndarray = None, loss: str = "lmse", optimizer: str = "L-BFGS-B", optimizer_options: dict = None,
            try_global: bool = False, global_options: dict = None, logger: logging.Logger = None,
            progress_callback: Callable[[float], None] = None,
            need_history: bool = True) -> Tuple[Optional[SSUResult], str]:
    assert isinstance(sample, (ArtificialSample, Sample))
    assert isinstance(distribution_type, DistributionType)
    assert isinstance(n_components, int)
    distribution_class = get_distribution(distribution_type)
    if x0 is None:
        x0 = distribution_class.get_defaults(n_components)
    else:
        x0 = np.array(x0)
        assert x0.ndim == 2
        assert x0.shape == (distribution_class.N_PARAMETERS + 1, n_components)
    check_loss(loss)
    if loss == "wasserstein":
        def loss_func(values, targets, _):
            return np.log(wasserstein_distance(sample.classes_phi, sample.classes_phi, values, targets))
    else:
        loss_func = loss_numpy(loss)
    check_optimizer(optimizer)
    if optimizer_options is not None:
        assert isinstance(optimizer_options, dict)
        if "maxiter" in optimizer_options:
            assert isinstance(optimizer_options["maxiter"], int)
            assert optimizer_options["maxiter"] > 0
    else:
        optimizer_options = dict(maxiter=200)

    if global_options is not None:
        assert isinstance(global_options, dict)
        if "niter" in global_options:
            assert isinstance(global_options["niter"], int)
            assert global_options["niter"] > 0
        if "niter_success" in global_options:
            assert isinstance(global_options["niter_success"], int)
            assert global_options["niter_success"] > 0
        if "stepsize" in global_options:
            assert isinstance(global_options["stepsize"], float)
            assert global_options["stepsize"] > 0
    else:
        global_options = dict(niter=100, niter_success=5, stepsize=0.2)

    if logger is None:
        logger = logging.getLogger("QGrain")
    else:
        assert isinstance(logger, logging.Logger)

    start_text = f"""Performing the SSU algorithm on the sample ({sample.name}).
    Distribution type: {distribution_type.name}
    Number of components: {n_components}
    x0: {x0 if x0 is None else x0.tolist()}
    Loss: {loss}
    Optimizer: {optimizer}
    Optimizer options: {optimizer_options}
    Try global optimization: {try_global}
    Global optimization: {global_options}
    Need history: {need_history}"""
    logger.debug(start_text)

    start_time = time.time()
    global_iteration = 0
    iteration = 0
    if try_global:
        max_iterations = global_options["niter"] * optimizer_options["maxiter"]
    else:
        max_iterations = optimizer_options["maxiter"]
    history = [np.expand_dims(x0, axis=0)]
    classes = np.expand_dims(np.expand_dims(sample.classes_phi, 0), 0).repeat(n_components, 1)

    def closure(x):
        x = x.reshape((1, distribution_class.N_PARAMETERS + 1, n_components))
        proportions, components, _ = distribution_class.interpret(x, classes, sample.interval_phi)
        pred_distribution = (proportions[0] @ components[0]).squeeze()
        loss_value = loss_func(np.clip(pred_distribution, 1e-8, 1.0),
                               np.clip(sample.distribution, 1e-8, 1.0), None)
        return loss_value

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

    if try_global:
        global_result = basinhopping(
            closure, x0=x0.reshape(-1), callback=global_callback,
            minimizer_kwargs=dict(method=optimizer, callback=callback, options=optimizer_options),
            **global_options)
        if global_result.lowest_optimization_result.success or global_result.lowest_optimization_result.status == 9:
            parameters = np.reshape(global_result.x, (1, distribution_class.N_PARAMETERS + 1, n_components))
            message = global_result.message
            if need_history:
                history.append(parameters)
        else:
            logger.error(f"The fitting process terminated with a error: {global_result.message}.")
            return None, global_result.message
    else:
        local_result = minimize(closure, x0=x0.reshape(-1), method=optimizer,
                                callback=callback, options=optimizer_options)
        if local_result.success or local_result.status == 9:
            parameters = np.reshape(local_result.x, (1, distribution_class.N_PARAMETERS + 1, n_components))
            message = local_result.message
        else:
            logger.error(f"The fitting process terminated with a error: {local_result.message}.")
            return None, local_result.message

    time_spent = time.time() - start_time
    if need_history:
        parameters = np.concatenate(history, axis=0)
    # sort the components by their grain sizes (from fine to coarse)
    sorted_parameters = sort_components(distribution_type, parameters, mode="last")
    settings = dict(loss=loss, optimizer=optimizer, try_global=try_global,
                    global_options=global_options, need_history=need_history)
    ssu_result = SSUResult(sample, distribution_type, sorted_parameters, time_spent, x0=x0, settings=settings)
    logger.debug(f"The fitting process successfully finished. {message}")
    return ssu_result, message


def _execute(args: Tuple[Union[ArtificialSample, Sample], DistributionType, int, Dict[str, Any]]):
    sample, distribution_type, n_components, options = args
    if options is None:
        options = {}
    return try_ssu(sample, distribution_type, n_components, **options)


def try_dataset(
        dataset: Union[ArtificialDataset, Dataset],
        distribution_type: DistributionType,
        n_components: int,
        n_processes: int = 1,
        options: Dict[str, Any] = None):
    multiprocessing.freeze_support()
    with multiprocessing.Pool(n_processes) as pool:
        args = [(sample, distribution_type, n_components, options) for sample in dataset]
        results = pool.map(_execute, args)
        succeeded_results: List[SSUResult] = []
        failed_samples: List[Tuple[int, str]] = []
        for i, (result, message) in enumerate(results):
            if isinstance(result, SSUResult):
                succeeded_results.append(result)
            else:
                failed_samples.append((i, message))
    return succeeded_results, failed_samples
