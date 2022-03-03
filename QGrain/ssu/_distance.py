import numpy as np


def log10MSE_distance(values: np.ndarray, targets: np.ndarray, axis=None) -> float:
    return np.log10(np.mean(np.square(values - targets), axis=axis))

def MSE_distance(values: np.ndarray, targets: np.ndarray, axis=None) -> float:
    return np.mean(np.square(values - targets), axis=axis)

def p_norm(values: np.ndarray, targets: np.ndarray, p=2, axis=None) -> float:
    return np.sum(np.abs(values - targets) ** p, axis=axis) ** (1 / p)

def cosine_distance(values: np.ndarray, targets: np.ndarray, axis=None) -> float:
    if np.all(np.equal(values, 0.0)) or np.all(np.equal(targets, 0.0)):
        return 1.0
    cosine = np.sum(values * targets, axis=axis) / (np.sqrt(np.sum(np.square(values), axis=axis)) * np.sqrt(np.sum(np.square(targets), axis=axis)))
    return abs(cosine)

def angular_distance(values: np.ndarray, targets: np.ndarray, axis=None) -> float:
    cosine = cosine_distance(values, targets, axis=axis)
    angular = 2 * np.arccos(cosine) / np.pi
    return angular

def get_distance_func_by_name(distance: str):
    if distance[-4:] == "norm":
        p = int(distance[0])
        return lambda x, y, axis=None: p_norm(x, y, p, axis=axis)
    elif distance == "MSE":
        return lambda x, y, axis=None: MSE_distance(x, y, axis=axis)
    elif distance == "log10MSE":
        return lambda x, y, axis=None: log10MSE_distance(x, y, axis=axis)
    elif distance == "cosine":
        return lambda x, y, axis=None: cosine_distance(x, y, axis=axis)
    elif distance == "angular":
        return lambda x, y, axis=None: angular_distance(x, y, axis=axis)
    else:
        raise NotImplementedError(distance)
