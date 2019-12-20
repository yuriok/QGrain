from typing import Dict, List, Tuple

import numpy as np
import uuid

class SampleData:
    __slots__ = "name", "distribution"
    def __init__(self, name, distribution: np.ndarray):
        self.name = name
        self.distribution = distribution


class GrainSizeData:
    __slots__ = "is_valid", "classes", "sample_data_list"
    def __init__(self, is_valid = False, classes: np.ndarray = None, sample_data_list: List[SampleData] = None):
        self.is_valid = is_valid
        self.classes = classes
        self.sample_data_list = sample_data_list


class FittedData:
    __slots__ = "name", "target", "sum", "mse", "components", "statistic", "uuid"
    def __init__(self, name: str, target: Tuple[np.ndarray, np.ndarray],
                 sum_data: Tuple[np.ndarray, np.ndarray], mse: float,
                 components: List[Tuple[np.ndarray, np.ndarray]],
                 statistic: List[Dict]):
        self.name = name
        self.target = target
        self.sum = sum_data
        self.mse = mse  # Mean Squared Error
        self.components = components
        self.statistic = statistic
        # add uuid to manage data
        self.uuid = uuid.uuid4()

    def has_invalid_value(self) -> bool:
        if self.name is None or self.name == "":
            return True
        if self.mse is np.nan:
            return True
        if np.any(np.isnan(self.target[0])):
            return True
        if np.any(np.isnan(self.target[1])):
            return True
        if np.any(np.isnan(self.sum[0])):
            return True
        if np.any(np.isnan(self.sum[1])):
            return True
        for comp in self.components:
            if np.any(np.isnan(comp[0])):
                return True
            if np.any(np.isnan(comp[1])):
                return True
        for i, comp in enumerate(self.statistic):
            for name, value in comp.items():
                if np.isreal(value) and np.isnan(value):
                    return True
        return False