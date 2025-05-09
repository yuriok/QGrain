__all__ = ["loss_torch", "NonparametricKernel", "NormalKernel", "SkewNormalKernel", "WeibullKernel",
           "GeneralWeibullKernel", "ProportionModule", "get_kernel"]

from typing import *

import numpy as np
import torch
from torch import Tensor

from .models import KernelType

_INFINITESIMAL = 1e-8


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


def normal_pdf(x, loc, scale):
    pdf = 1 / (scale * np.sqrt(2 * np.pi)) * torch.exp(-torch.square(x - loc) / (2 * scale ** 2))
    return pdf


def std_normal_cdf(x):
    cdf = 0.5 * (1 + torch.erf(x / np.sqrt(2)))
    return cdf


def skew_normal_pdf(x, shape, loc, scale):
    pdf = 2 * normal_pdf(x, loc, scale) * std_normal_cdf(shape * (x - loc) / scale)
    return pdf


def weibull_pdf(x, shape, scale):
    key = torch.greater_equal(x, 0)
    pdf = torch.zeros_like(x)
    pdf[key] = (shape[key] / scale[key]) * (x[key] / scale[key]) ** (
            shape[key] - 1) * torch.exp(-(x[key] / scale[key]) ** shape[key])
    return pdf


def general_weibull_pdf(x, shape, loc, scale):
    y = x - loc
    key = torch.greater_equal(y, 0)
    pdf = torch.zeros_like(y)
    pdf[key] = (shape[key] / scale[key]) * (y[key] / scale[key]) ** (
            shape[key] - 1) * torch.exp(-(y[key] / scale[key]) ** shape[key])
    return pdf


class NonparametricKernel(torch.nn.Module):
    N_PARAMETERS = None

    def __init__(self, n_samples: int, n_members: int, n_classes: int, parameters: np.ndarray = None):
        super().__init__()
        self.N_PARAMETERS = n_classes
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        if parameters is None:
            self._params = torch.nn.Parameter(torch.randn(n_samples, n_members, n_classes) + 2.0, requires_grad=True)
        else:
            assert isinstance(parameters, np.ndarray)
            if parameters.ndim == 2:
                assert parameters.shape == (n_members, n_classes)
                self._params = torch.nn.Parameter(
                    torch.from_numpy(np.expand_dims(parameters, 0).repeat(n_samples, 0)), requires_grad=True)
            else:
                assert parameters.ndim == 3
                assert parameters.shape == (n_samples, n_members, n_classes)
                self._params = torch.nn.Parameter(torch.from_numpy(parameters), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: torch.Tensor):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        end_members = torch.softmax(self._params, dim=-1)
        return end_members


class NormalKernel(torch.nn.Module):
    N_PARAMETERS = 2

    def __init__(self, n_samples: int, n_members: int, n_classes: int, parameters: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        if parameters is None:
            self._params = torch.nn.Parameter(
                torch.randn(n_samples, self.N_PARAMETERS, n_members) + 5.0, requires_grad=True)
        else:
            assert isinstance(parameters, np.ndarray)
            if parameters.ndim == 2:
                assert parameters.shape == (self.N_PARAMETERS, n_members)
                self._params = torch.nn.Parameter(
                    torch.from_numpy(np.expand_dims(parameters, 0).repeat(n_samples, 0)), requires_grad=True)
            else:
                assert parameters.ndim == 3
                assert parameters.shape == (n_samples, self.N_PARAMETERS, n_members)
                self._params = torch.nn.Parameter(torch.from_numpy(parameters), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: torch.Tensor):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        locations = self._params[:, 0, :].unsqueeze(2).repeat(1, 1, self.n_classes)
        scales = (torch.relu(self._params[:, 1, :]) + _INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        end_members = normal_pdf(classes, locations, scales) * interval
        return end_members


class SkewNormalKernel(torch.nn.Module):
    N_PARAMETERS = 3

    def __init__(self, n_samples: int, n_members: int, n_classes: int, parameters: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        if parameters is None:
            self._params = torch.nn.Parameter(
                torch.randn(n_samples, self.N_PARAMETERS, n_members) + 5.0, requires_grad=True)
        else:
            assert isinstance(parameters, np.ndarray)
            if parameters.ndim == 2:
                assert parameters.shape == (self.N_PARAMETERS, n_members)
                self._params = torch.nn.Parameter(
                    torch.from_numpy(np.expand_dims(parameters, 0).repeat(n_samples, 0)), requires_grad=True)
            else:
                assert parameters.ndim == 3
                assert parameters.shape == (n_samples, self.N_PARAMETERS, n_members)
                self._params = torch.nn.Parameter(torch.from_numpy(parameters), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: torch.Tensor):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        shapes = self._params[:, 0, :].unsqueeze(2).repeat(1, 1, self.n_classes)
        locations = self._params[:, 1, :].unsqueeze(2).repeat(1, 1, self.n_classes)
        scales = (torch.relu(self._params[:, 2, :]) + _INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        end_members = skew_normal_pdf(classes, shapes, locations, scales) * interval
        return end_members


class WeibullKernel(torch.nn.Module):
    N_PARAMETERS = 2

    def __init__(self, n_samples: int, n_members: int, n_classes: int, parameters: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        if parameters is None:
            self._params = torch.nn.Parameter(
                torch.randn(n_samples, self.N_PARAMETERS, n_members) + 5.0, requires_grad=True)
        else:
            assert isinstance(parameters, np.ndarray)
            if parameters.ndim == 2:
                assert parameters.shape == (self.N_PARAMETERS, n_members)
                self._params = torch.nn.Parameter(
                    torch.from_numpy(np.expand_dims(parameters, 0).repeat(n_samples, 0)), requires_grad=True)
            else:
                assert parameters.ndim == 3
                assert parameters.shape == (n_samples, self.N_PARAMETERS, n_members)
                self._params = torch.nn.Parameter(torch.from_numpy(parameters), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: torch.Tensor):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        shapes = (torch.relu(self._params[:, 0, :]) + _INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        scales = (torch.relu(self._params[:, 1, :]) + _INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        end_members = weibull_pdf(classes, shapes, scales) * interval
        return end_members


class GeneralWeibullKernel(torch.nn.Module):
    N_PARAMETERS = 3

    def __init__(self, n_samples: int, n_members: int, n_classes: int, parameters: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes

        if parameters is None:
            self._params = torch.nn.Parameter(
                torch.randn(n_samples, self.N_PARAMETERS, n_members) + 5.0, requires_grad=True)
        else:
            assert isinstance(parameters, np.ndarray)
            if parameters.ndim == 2:
                assert parameters.shape == (self.N_PARAMETERS, n_members)
                self._params = torch.nn.Parameter(
                    torch.from_numpy(np.expand_dims(parameters, 0).repeat(n_samples, 0)), requires_grad=True)
            else:
                assert parameters.ndim == 3
                assert parameters.shape == (n_samples, self.N_PARAMETERS, n_members)
                self._params = torch.nn.Parameter(torch.from_numpy(parameters), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: torch.Tensor):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        shapes = (torch.relu(self._params[:, 0, :]) + _INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        locations = self._params[:, 1, :].unsqueeze(2).repeat(1, 1, self.n_classes)
        scales = (torch.relu(self._params[:, 2, :]) + _INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        end_members = general_weibull_pdf(classes, shapes, locations, scales) * interval
        return end_members


class ProportionModule(torch.nn.Module):
    def __init__(self, n_samples: int, n_members: int, parameters: np.ndarray = None):
        super().__init__()
        if parameters is None:
            self._params = torch.nn.Parameter(torch.rand(n_samples, 1, n_members), requires_grad=True)
        else:
            if parameters.ndim == 2:
                assert parameters.shape == (1, n_members)
                self._params = torch.nn.Parameter(
                    torch.from_numpy(np.expand_dims(parameters, 0).repeat(n_samples, 0)), requires_grad=True)
            else:
                assert parameters.ndim == 3
                assert parameters.shape == (n_samples, 1, n_members)
                self._params = torch.nn.Parameter(torch.from_numpy(parameters), requires_grad=True)

    def forward(self):
        # n_samples x 1 x n_members
        proportions = torch.softmax(self._params, 2)
        return proportions


def get_kernel(kernel_type: KernelType, n_samples: int, n_members: int, n_classes: int, parameters: np.ndarray = None):
    if kernel_type == KernelType.Nonparametric:
        return NonparametricKernel(n_samples, n_members, n_classes, parameters)
    elif kernel_type == KernelType.Normal:
        return NormalKernel(n_samples, n_members, n_classes, parameters)
    elif kernel_type == KernelType.SkewNormal:
        return SkewNormalKernel(n_samples, n_members, n_classes, parameters)
    elif kernel_type == KernelType.Weibull:
        return NormalKernel(n_samples, n_members, n_classes, parameters)
    elif kernel_type == KernelType.GeneralWeibull:
        return GeneralWeibullKernel(n_samples, n_members, n_classes, parameters)
    else:
        raise NotImplementedError(kernel_type)
