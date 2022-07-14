from typing import *

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


# P-Norm
def p_norm_numpy(values: ndarray, targets: ndarray, p=2, axis=None) -> ndarray:
    return np.sum(np.abs(values - targets) ** p, axis=axis) ** (1 / p)


# Mean Absolute Error
def mae_numpy(value: ndarray, targets: ndarray, axis=None) -> ndarray:
    return np.mean(np.abs(value - targets), axis=axis)


# Mean Squared Error
def mse_numpy(values: ndarray, targets: ndarray, axis=None) -> ndarray:
    return np.mean(np.square(values - targets), axis=axis)


# Root Mean Squared Error
def rmse_numpy(values: ndarray, targets: ndarray, axis=None) -> ndarray:
    return np.sqrt(np.mean(np.square(values - targets), axis=axis))


# Root Mean Squared Logarithmic Error
def rmlse_numpy(values: ndarray, targets: ndarray, axis=None) -> ndarray:
    return np.sqrt(np.mean(np.square(np.log(values + 1) - np.log(targets + 1)), axis=axis))


# Logarithmic Mean Squared Error
def lmse_numpy(values: ndarray, targets: ndarray, axis=None) -> ndarray:
    return np.log(np.mean(np.square(values - targets), axis=axis))


# Cosine
def cosine_numpy(values: ndarray, targets: ndarray, axis=None) -> ndarray:
    if np.all(np.equal(values, 0.0)) or np.all(np.equal(targets, 0.0)):
        return 1.0
    return np.sum(values * targets, axis=axis) / (
            np.sqrt(np.sum(np.square(values), axis=axis)) * np.sqrt(np.sum(np.square(targets), axis=axis)))


# Angular
def angular_numpy(values: ndarray, targets: ndarray, axis=None) -> ndarray:
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


# P-Norm
def p_norm_torch(values: Tensor, targets: Tensor, p=2) -> Tensor:
    return torch.sum(torch.abs(values - targets) ** p) ** (1 / p)


# Mean Absolute Error
def mae_torch(values: Tensor, targets: Tensor) -> Tensor:
    return torch.mean(torch.abs(values - targets))


# Mean Squared Error
def mse_torch(values: Tensor, targets: Tensor) -> Tensor:
    return torch.mean(torch.square(values - targets))


# Root Mean Squared Error
def rmse_torch(values: Tensor, targets: Tensor) -> Tensor:
    return torch.sqrt(torch.mean(torch.square(values - targets)))


# Root Mean Squared Logarithmic Error
def rmlse_torch(values: Tensor, targets: Tensor) -> Tensor:
    return torch.sqrt(torch.mean(torch.square(torch.log(values + 1) - torch.log(targets + 1))))


# Logarithmic Mean Squared Error
def lmse_torch(values: Tensor, targets: Tensor) -> Tensor:
    return torch.log(torch.mean(torch.square(values - targets)))


# Cosine
def cosine_torch(values: Tensor, targets: Tensor) -> Tensor:
    return torch.sum(values * targets) / (
            torch.sqrt(torch.sum(torch.square(values))) * torch.sqrt(torch.sum(torch.square(targets))))


# Angular
def angular_torch(values: Tensor, targets: Tensor) -> Tensor:
    return 2 * torch.arccos(cosine_torch(values, targets)) / np.pi


def loss_torch(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    if name[-4:] == "norm":
        p = int(name[:-5])
        return lambda x, y: p_norm_torch(x, y, p)
    elif name == "mae":
        return lambda x, y: mae_torch(x, y)
    elif name == "mse":
        return lambda x, y: mse_torch(x, y)
    elif name == "rmse":
        return lambda x, y: rmse_torch(x, y)
    elif name == "rmlse":
        return lambda x, y: rmlse_torch(x, y)
    elif name == "lmse":
        return lambda x, y: lmse_torch(x, y)
    elif name == "cosine":
        return lambda x, y: cosine_torch(x, y)
    elif name == "angular":
        return lambda x, y: angular_torch(x, y)
    else:
        raise NotImplementedError(name)
