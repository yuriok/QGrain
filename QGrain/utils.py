__all__ = ["get_image_by_proportions", "udm_to_ssu"]

import logging
from typing import *

import numpy as np
from numpy import ndarray

from QGrain.models import DistributionType, UDMResult, SSUResult


def get_image_by_proportions(proportions: ndarray, resolution: int = 100) -> ndarray:
    n_samples, n_components = proportions.shape
    index = np.repeat(np.expand_dims(np.linspace(0.0, 1.0, resolution), axis=0), n_samples, axis=0)
    image = np.zeros((n_samples, resolution))

    bound = np.ones((n_samples, n_components+1))
    bound[:, 0] = 0.0
    bottom = np.zeros(n_samples)
    for i in range(n_components):
        bottom += proportions[:, i]
        bound[:, i+1] = bottom
    bound[:, -1] = 1.0

    for i in range(n_components):
        lower = np.repeat(np.expand_dims(bound[:, i], axis=1), resolution, axis=1)
        upper = np.repeat(np.expand_dims(bound[:, i+1], axis=1), resolution, axis=1)
        key = np.logical_and(np.greater_equal(index, lower), np.less_equal(index, upper))
        image[key] = i
    return image.T


def udm_to_ssu(result: UDMResult, logger: logging.Logger = None,
               progress_callback: Callable[[float], None] = None) -> List[SSUResult]:
    assert isinstance(result, UDMResult)
    if logger is None:
        logger = logging.getLogger("QGrain")
    else:
        assert isinstance(logger, logging.Logger)
    distribution_type = DistributionType.__members__[result.kernel_type.name]
    weight = np.ones((1, result.n_components))
    x0 = np.concatenate([result.x0, weight], axis=0).astype(np.float32)
    time_spent = result.time_spent / result.n_samples
    ssu_results = []
    for i in range(result.n_samples):
        if result.n_iterations == 0:
            parameters = np.expand_dims(result.parameters[-1, i], axis=0)
        else:
            history = [np.expand_dims(result.parameters[j][i], axis=0) for j in range(result.n_iterations)]
            parameters = np.concatenate(history, axis=0)
        ssu_result = SSUResult(result.dataset[i], distribution_type, parameters, time_spent,
                               x0=x0, settings=result.settings)
        ssu_results.append(ssu_result)
        if progress_callback is not None:
            progress_callback(i / result.n_samples)
    if progress_callback is not None:
        progress_callback(1.0)
    return ssu_results
