from typing import Dict, List, Tuple

import numpy as np


class SampleData:
    def __init__(self, name, distribution: np.ndarray):
        self.name = name
        self.distribution = distribution


class GrainSizeData:
    def __init__(self, is_valid = False, classes: np.ndarray = None, sample_data_list: List[SampleData] = None):
        self.is_valid = is_valid
        self.classes = classes
        self.sample_data_list = sample_data_list


class FittedData:

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
