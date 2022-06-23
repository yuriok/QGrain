from enum import Enum, unique

import numpy as np
import torch

_INFINITESIMAL = 1e-8


@unique
class KernelType(Enum):
    Nonparametric = "Nonparametric"
    Normal = "Normal"
    SkewNormal = "Skew Normal"
    Weibull = "Weibull"
    GeneralWeibull = "General Weibull"


def normal_pdf(x, loc, scale):
    pdf = 1 / (scale*np.sqrt(2*np.pi)) * torch.exp(-torch.square(x - loc) / (2*scale**2))
    return pdf


def std_normal_cdf(x):
    cdf = 0.5 * (1 + torch.erf(x / np.sqrt(2)))
    return cdf


def skew_normal_pdf(x, shape, loc, scale):
    pdf = 2 * normal_pdf(x, loc, scale) * std_normal_cdf(shape*(x-loc)/scale)
    return pdf


def weibull_pdf(x, shape, scale):
    key = torch.greater_equal(x, 0)
    pdf = torch.zeros_like(x)
    pdf[key] = (shape[key] / scale[key]) * (x[key] / scale[key]) ** (shape[key]-1) * torch.exp(-(x[key]/scale[key])**shape[key])
    return pdf


def general_weibull_pdf(x, shape, loc, scale):
    y = x - loc
    key = torch.greater_equal(y, 0)
    pdf = torch.zeros_like(y)
    pdf[key] = (shape[key] / scale[key]) * (y[key] / scale[key]) ** (shape[key]-1) * torch.exp(-(y[key]/scale[key])**shape[key])
    return pdf


class NonparametricKernel(torch.nn.Module):
    def __init__(self, n_samples: int, n_members: int, n_classes: int, params: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        if params is None:
            self.params = torch.nn.Parameter(torch.randn(n_samples, n_members, n_classes)+2.0, requires_grad=True)
        else:
            assert params.ndim == 2
            assert params.shape == (n_members, n_classes)
            self.params = torch.nn.Parameter(torch.from_numpy(np.expand_dims(params, 0).repeat(n_samples, 0)), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: float):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        end_members = torch.softmax(self.params, dim=-1)
        return end_members

class NormalKernel(torch.nn.Module):
    N_PARAMS = 2
    def __init__(self, n_samples: int, n_members: int, n_classes: int, params: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        if params is None:
            self.params = torch.nn.Parameter(torch.randn(n_samples, self.N_PARAMS, n_members)+5.0, requires_grad=True)
        else:
            assert params.ndim == 2
            assert params.shape == (self.N_PARAMS, n_members)
            self.params = torch.nn.Parameter(torch.from_numpy(np.expand_dims(params, 0).repeat(n_samples, 0)), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: float):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        locations = self.params[:, 0, :].unsqueeze(2).repeat(1, 1, self.n_classes)
        scales = (torch.relu(self.params[:, 1, :])+_INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        end_members = normal_pdf(classes, locations, scales) * interval
        # params = torch.cat([locations[:, :, 0].unsqueeze(1), scales[:, :, 0].unsqueeze(1)], dim=1)
        return end_members


class SkewNormalKernel(torch.nn.Module):
    N_PARAMS = 3
    def __init__(self, n_samples: int, n_members: int, n_classes: int, params: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        if params is None:
            self.params = torch.nn.Parameter(torch.randn(n_samples, self.N_PARAMS, n_members)+5.0, requires_grad=True)
        else:
            assert params.ndim == 2
            assert params.shape == (self.N_PARAMS, n_members)
            self.params = torch.nn.Parameter(torch.from_numpy(np.expand_dims(params, 0).repeat(n_samples, 0)), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: float):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        shapes = self.params[:, 0, :].unsqueeze(2).repeat(1, 1, self.n_classes)
        locations = self.params[:, 1, :].unsqueeze(2).repeat(1, 1, self.n_classes)
        scales = (torch.relu(self.params[:, 2, :])+_INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        end_members = skew_normal_pdf(classes, shapes, locations, scales) * interval
        # params = torch.cat([shapes[:, :, 0].unsqueeze(1), locations[:, :, 0].unsqueeze(1), scales[:, :, 0].unsqueeze(1)], dim=1)
        return end_members


class WeibullKernel(torch.nn.Module):
    N_PARAMS = 2
    def __init__(self, n_samples: int, n_members: int, n_classes: int, params: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        if params is None:
            self.params = torch.nn.Parameter(torch.randn(n_samples, self.N_PARAMS, n_members)+5.0, requires_grad=True)
        else:
            assert params.ndim == 2
            assert params.shape == (self.N_PARAMS, n_members)
            self.params = torch.nn.Parameter(torch.from_numpy(np.expand_dims(params, 0).repeat(n_samples, 0)), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: float):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        shapes = (torch.relu(self.params[:, 0, :])+_INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        scales = (torch.relu(self.params[:, 1, :])+_INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        end_members = weibull_pdf(classes, shapes, scales) * interval
        # params = torch.cat([shapes[:, :, 0].unsqueeze(1), scales[:, :, 0].unsqueeze(1)], dim=1)
        return end_members


class GeneralWeibullKernel(torch.nn.Module):
    N_PARAMS = 3
    def __init__(self, n_samples: int, n_members: int, n_classes: int, params: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes

        if params is None:
            self.params = torch.nn.Parameter(torch.randn(n_samples, self.N_PARAMS, n_members)+5.0, requires_grad=True)
        else:
            assert params.ndim == 2
            assert params.shape == (self.N_PARAMS, n_members)
            self.params = torch.nn.Parameter(torch.from_numpy(np.expand_dims(params, 0).repeat(n_samples, 0)), requires_grad=True)

    def forward(self, classes: torch.Tensor, interval: float):
        assert classes.shape == (self.n_samples, self.n_members, self.n_classes)
        shapes = (torch.relu(self.params[:, 0, :])+_INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        locations = self.params[:, 1, :].unsqueeze(2).repeat(1, 1, self.n_classes)
        scales = (torch.relu(self.params[:, 2, :])+_INFINITESIMAL).unsqueeze(2).repeat(1, 1, self.n_classes)
        end_members = general_weibull_pdf(classes, shapes, locations, scales) * interval
        # params = torch.cat([shapes[:, :, 0].unsqueeze(1), locations[:, :, 0].unsqueeze(1), scales[:, :, 0].unsqueeze(1)], dim=1)
        return end_members

class Proportion(torch.nn.Module):
    def __init__(self, n_samples: int, n_members: int):
        super().__init__()
        self.params = torch.nn.Parameter(torch.rand(n_samples, 1, n_members), requires_grad=True)

    def forward(self):
        # n_samples x 1 x n_members
        proportions = torch.softmax(self.params, 2)
        return proportions


KERNEL_CLASS_MAP = {
    KernelType.Nonparametric: NonparametricKernel,
    KernelType.Normal: NormalKernel,
    KernelType.SkewNormal: SkewNormalKernel,
    KernelType.Weibull: WeibullKernel,
    KernelType.GeneralWeibull: GeneralWeibullKernel}
