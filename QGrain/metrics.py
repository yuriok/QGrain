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
    return np.sqrt(np.mean(np.square(np.log(values + 1) - np.log(targets + 1))))


# Cosine
def cosine_numpy(values: ndarray, targets: ndarray, axis=None) -> ndarray:
    if np.all(np.equal(values, 0.0)) or np.all(np.equal(targets, 0.0)):
        return 1.0
    return np.sum(values * targets, axis=axis) / (
            np.sqrt(np.sum(np.square(values), axis=axis)) * np.sqrt(np.sum(np.square(targets), axis=axis)))


# Angular
def angular_numpy(values: ndarray, targets: ndarray, axis=None) -> ndarray:
    return 2 * np.arccos(cosine_numpy(values, targets, axis=axis)) / np.pi


def loss_numpy(distance: str) -> Callable[[ndarray, ndarray, Optional[int]], ndarray]:
    if distance[-4:] == "norm":
        p = int(distance[0])
        return lambda x, y, axis=None: p_norm_numpy(x, y, p, axis=axis)
    elif distance == "mae":
        return lambda x, y, axis=None: mae_numpy(x, y, axis=axis)
    elif distance == "mse":
        return lambda x, y, axis=None: mse_numpy(x, y, axis=axis)
    elif distance == "rmse":
        return lambda x, y, axis=None: rmse_numpy(x, y, axis=axis)
    elif distance == "rmlse":
        return lambda x, y, axis=None: rmlse_numpy(x, y, axis=axis)
    elif distance == "cosine":
        return lambda x, y, axis=None: cosine_numpy(x, y, axis=axis)
    elif distance == "angular":
        return lambda x, y, axis=None: angular_numpy(x, y, axis=axis)
    else:
        raise NotImplementedError(distance)


# P-Norm
def p_norm_torch(values: Tensor, targets: Tensor, p=2, dim=None) -> Tensor:
    return torch.sum(torch.abs(values - targets) ** p, dim=dim) ** (1 / p)


# Mean Absolute Error
def mae_torch(values: Tensor, targets: Tensor, dim=None) -> Tensor:
    return torch.mean(torch.abs(values - targets), dim=dim)


# Mean Squared Error
def mse_torch(values: Tensor, targets: Tensor, dim=None) -> Tensor:
    return torch.mean(torch.square(values - targets), dim=dim)


# Root Mean Squared Error
def rmse_torch(values: Tensor, targets: Tensor, dim=None) -> Tensor:
    return torch.sqrt(torch.mean(torch.square(values - targets), dim=dim))


# Root Mean Squared Logarithmic Error
def rmlse_torch(values: Tensor, targets: Tensor, dim=None) -> Tensor:
    return torch.sqrt(torch.mean(torch.square(torch.log(values + 1) - torch.log(targets + 1)), dim=dim))


# Cosine
def cosine_torch(values: Tensor, targets: Tensor, dim=None) -> Tensor:
    return torch.sum(values * targets, dim=dim) / (
            torch.sqrt(torch.sum(torch.square(values), dim=dim)) * torch.sqrt(
                torch.sum(torch.square(targets), dim=dim)))


# Angular
def angular_torch(values: Tensor, targets: Tensor, dim=None) -> Tensor:
    return 2 * torch.arccos(cosine_torch(values, targets, dim=dim)) / np.pi


def loss_torch(distance: str) -> Callable[[Tensor, Tensor, Optional[int]], Tensor]:
    if distance[-4:] == "norm":
        p = int(distance[0])
        return lambda x, y, dim=None: p_norm_torch(x, y, p, dim=dim)
    elif distance == "mae":
        return lambda x, y, dim=None: mae_torch(x, y, dim=dim)
    elif distance == "mse":
        return lambda x, y, dim=None: mse_torch(x, y, dim=dim)
    elif distance == "rmse":
        return lambda x, y, dim=None: rmse_torch(x, y, dim=dim)
    elif distance == "rmlse":
        return lambda x, y, dim=None: rmlse_torch(x, y, dim=dim)
    elif distance == "cosine":
        return lambda x, y, dim=None: cosine_torch(x, y, dim=dim)
    elif distance == "angular":
        return lambda x, y, dim=None: angular_torch(x, y, dim=dim)
    else:
        raise NotImplementedError(distance)

