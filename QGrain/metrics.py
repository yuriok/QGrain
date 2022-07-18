from typing import *

import numpy as np
from numpy import ndarray


# P-Norm
def p_norm_numpy(values: ndarray, targets: ndarray, p=2, axis: Union[int, Sequence[int]] = None) -> ndarray:
    return np.sum(np.abs(values - targets) ** p, axis=axis) ** (1 / p)


# Mean Absolute Error
def mae_numpy(value: ndarray, targets: ndarray, axis: Union[int, Sequence[int]] = None) -> ndarray:
    return np.mean(np.abs(value - targets), axis=axis)


# Mean Squared Error
def mse_numpy(values: ndarray, targets: ndarray, axis: Union[int, Sequence[int]] = None) -> ndarray:
    return np.mean(np.square(values - targets), axis=axis)


# Root Mean Squared Error
def rmse_numpy(values: ndarray, targets: ndarray, axis: Union[int, Sequence[int]] = None) -> ndarray:
    return np.sqrt(np.mean(np.square(values - targets), axis=axis))


# Root Mean Squared Logarithmic Error
def rmlse_numpy(values: ndarray, targets: ndarray, axis: Union[int, Sequence[int]] = None) -> ndarray:
    return np.sqrt(np.mean(np.square(np.log(values + 1) - np.log(targets + 1)), axis=axis))


# Logarithmic Mean Squared Error
def lmse_numpy(values: ndarray, targets: ndarray, axis: Union[int, Sequence[int]] = None) -> ndarray:
    return np.log(np.mean(np.square(values - targets), axis=axis))


# Cosine
def cosine_numpy(values: ndarray, targets: ndarray, axis: Union[int, Sequence[int]] = None) -> ndarray:
    if np.all(np.equal(values, 0.0)) or np.all(np.equal(targets, 0.0)):
        return 1.0
    return np.sum(values * targets, axis=axis) / (
            np.sqrt(np.sum(np.square(values), axis=axis)) * np.sqrt(np.sum(np.square(targets), axis=axis)))


# Angular
def angular_numpy(values: ndarray, targets: ndarray, axis: Union[int, Sequence[int]] = None) -> ndarray:
    return 2 * np.arccos(cosine_numpy(values, targets, axis=axis)) / np.pi


def loss_numpy(name: str) -> Callable[[ndarray, ndarray, Optional[int]], Union[float, ndarray]]:
    if name[-4:] == "norm":
        p = int(name[:-5])
        return lambda x, y, axis=None: p_norm_numpy(x, y, p, axis=axis)
    elif name == "mae":
        return lambda x, y, axis=None: mae_numpy(x, y, axis=axis)
    elif name == "mse":
        return lambda x, y, axis=None: mse_numpy(x, y, axis=axis)
    elif name == "rmse":
        return lambda x, y, axis=None: rmse_numpy(x, y, axis=axis)
    elif name == "rmlse":
        return lambda x, y, axis=None: rmlse_numpy(x, y, axis=axis)
    elif name == "lmse":
        return lambda x, y, axis=None: lmse_numpy(x, y, axis=axis)
    elif name == "cosine":
        return lambda x, y, axis=None: cosine_numpy(x, y, axis=axis)
    elif name == "angular":
        return lambda x, y, axis=None: angular_numpy(x, y, axis=axis)
    else:
        raise NotImplementedError(name)
