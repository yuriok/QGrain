import numpy as np


def log10MSE_distance(values: np.ndarray, targets: np.ndarray) -> float:
    return np.log10(np.mean(np.square(values - targets)))

def MSE_distance(values: np.ndarray, targets: np.ndarray) -> float:
    return np.mean(np.square(values - targets))

def p_norm(values: np.ndarray, targets: np.ndarray, p=2) -> float:
    return np.sum(np.abs(values - targets) ** p) ** (1 / p)

def cosine_distance(values: np.ndarray, targets: np.ndarray) -> float:
    if np.all(np.equal(values, 0.0)) or np.all(np.equal(targets, 0.0)):
        return 1.0
    cosine = np.sum(values * targets) / (np.sqrt(np.sum(np.square(values))) * np.sqrt(np.sum(np.square(targets))))
    return abs(cosine)

def angular_distance(values: np.ndarray, targets: np.ndarray) -> float:
    cosine = cosine_distance(values, targets)
    angular = 2 * np.arccos(cosine) / np.pi
    return angular

def get_distance_func_by_name(distance: str):
    if distance[-4:] == "norm":
        p = int(distance[0])
        return lambda x, y: p_norm(x, y, p)
    elif distance == "MSE":
        return lambda x, y: MSE_distance(x, y)
    elif distance == "log10MSE":
        return lambda x, y: log10MSE_distance(x, y)
    elif distance == "cosine":
        return lambda x, y: cosine_distance(x, y)
    elif distance == "angular":
        return lambda x, y: angular_distance(x, y)
    else:
        raise NotImplementedError(distance)
