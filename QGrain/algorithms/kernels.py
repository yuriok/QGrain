import numpy as np
import torch
from QGrain.algorithms import DistributionType
from QGrain.algorithms.distributions import BaseDistribution
from torch.nn import Module, Parameter, ReLU, Softmax


def normal_pdf(x, loc, scale):
    pdf = 1 / (scale*np.sqrt(2*np.pi)) * torch.exp(-torch.square(x - loc) / (2*scale**2))
    return pdf

def std_normal_cdf(x):
    cdf = 0.5 * (1 + torch.erf(x / np.sqrt(2)))
    return cdf

def weibull_pdf(x, shape, loc, scale):
    y = x - loc
    key = torch.greater_equal(y, 0)
    pdf = torch.zeros_like(y)
    pdf[key] = (shape / scale) * (y[key] / scale) ** (shape-1) * torch.exp(-(y[key]/scale)**shape)
    return pdf

def skew_normal_pdf(x, shape, loc, scale):
    pdf = 2 * normal_pdf(x, loc, scale) * std_normal_cdf(shape*(x-loc)/scale)
    return pdf

class NonparametricKernel(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.distribution = Parameter(torch.rand(n_classes), requires_grad=True)
        self.softmax = Softmax(dim=0)

    def forward(self, _):
        frequency = self.softmax(self.distribution)
        return frequency

    @property
    def frequency(self):
        with torch.no_grad():
            frequency = self.softmax(self.distribution)
            return frequency

class NormalKernel(Module):
    def __init__(self, loc=None, scale=None):
        super().__init__()
        self.__relu = ReLU()
        self.__loc = Parameter(torch.rand(1)+6 if loc is None else torch.Tensor([loc]), requires_grad=True)
        self.__scale = Parameter(torch.rand(1)+1 if scale is None else torch.Tensor([scale]), requires_grad=True)

    @property
    def loc(self) -> float:
        return self.__loc.item()

    @property
    def scale(self) -> float:
        with torch.no_grad():
            return self.__relu(self.__scale).item()

    @property
    def params(self):
        return self.loc, self.scale

    def forward(self, classes_φ):
        interval = torch.abs((classes_φ[0]-classes_φ[-1]) / (classes_φ.shape[0]-1)).item()
        loc = self.__loc
        scale = self.__relu(self.__scale)
        x = classes_φ
        pdf = normal_pdf(x, loc, scale)
        # scale pdf to frequency
        frequency = pdf * interval
        return frequency

class WeibullKernel(Module):
    def __init__(self, shape=None, loc=None, scale=None):
        super().__init__()
        self.__relu = ReLU()
        self.__shape = Parameter(torch.rand(1)+3 if shape is None else torch.Tensor([shape]), requires_grad=True)
        self.__loc = Parameter(torch.rand(1)+4 if loc is None else torch.Tensor([loc]), requires_grad=True)
        self.__scale = Parameter(torch.rand(1)+1 if scale is None else torch.Tensor([scale]), requires_grad=True)

    @property
    def shape(self) -> float:
        with torch.no_grad():
            return self.__relu(self.__shape).item()

    @property
    def loc(self) -> float:
        return self.__loc.item()

    @property
    def scale(self) -> float:
        with torch.no_grad():
            return self.__relu(self.__scale).item()

    @property
    def params(self):
        return self.shape, self.loc, self.scale

    def forward(self, classes_φ):
        interval = torch.abs((classes_φ[0]-classes_φ[-1]) / (classes_φ.shape[0]-1)).item()
        shape = self.__relu(self.__shape)
        loc = self.__loc
        scale = self.__relu(self.__scale)
        x = classes_φ
        pdf = weibull_pdf(x, shape, loc, scale)
        # scale pdf to frequency
        frequency = pdf * interval
        return frequency

class SkewNormalKernel(Module):
    def __init__(self, shape=None, loc=None, scale=None):
        super().__init__()
        self.__relu = ReLU()
        self.__shape = Parameter(torch.rand(1)*0.1 if shape is None else torch.Tensor([shape]), requires_grad=True)
        self.__loc = Parameter(torch.rand(1)+6 if loc is None else torch.Tensor([loc]), requires_grad=True)
        self.__scale = Parameter(torch.rand(1)+1 if scale is None else torch.Tensor([scale]), requires_grad=True)

    @property
    def shape(self) -> float:
        return self.__shape.item()

    @property
    def loc(self) -> float:
        return self.__loc.item()

    @property
    def scale(self) -> float:
        with torch.no_grad():
            return self.__relu(self.__scale).item()

    @property
    def params(self):
        return self.shape, self.loc, self.scale

    def forward(self, classes_φ):
        interval = torch.abs((classes_φ[0]-classes_φ[-1]) / (classes_φ.shape[0]-1)).item()
        shape = self.__shape
        loc = self.__loc
        scale = self.__relu(self.__scale)
        x = classes_φ
        pdf = skew_normal_pdf(x, shape, loc, scale)
        # scale pdf to frequency
        frequency = pdf * interval
        return frequency

def log10MSE_distance(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.log10(torch.mean(torch.square(values - targets)))

def MSE_distance(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(values - targets))

def p_norm(values: torch.Tensor, targets: torch.Tensor, p=2) -> torch.Tensor:
    return torch.sum(torch.abs(values - targets) ** p) ** (1 / p)

def cosine_distance(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    cosine = torch.sum(values * targets) / (torch.sqrt(torch.sum(torch.square(values))) * torch.sqrt(torch.sum(torch.square(targets))))
    return torch.abs(cosine)

def angular_distance(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    cosine = cosine_distance(values, targets)
    angular = 2 * torch.arccos(cosine) / np.pi
    return angular

def get_distance_func_by_name(distance: str):
    if distance[-4:] == "norm":
        p = int(distance[0])
        return lambda x, y: p_norm(x, y, p)
    elif distance == "MSE":
        return lambda x, y: MSE_distance(x, y)
    elif distance == "log10MSE":
        return lambda x, y: log10MSE_distance(x, y)
    elif distance == "angular":
        return lambda x, y: angular_distance(x, y)
    else:
        raise NotImplementedError(distance)

def get_initial_guess(distribution_type: DistributionType, reference):
    return BaseDistribution.get_initial_guess(distribution_type, reference)


KERNEL_MAP = {DistributionType.Normal: NormalKernel,
              DistributionType.Weibull: WeibullKernel,
              DistributionType.SkewNormal: SkewNormalKernel}

N_PARAMS_MAP = {DistributionType.Normal: 2,
                DistributionType.Weibull: 3,
                DistributionType.SkewNormal: 3}
