import numpy as np
import torch


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
