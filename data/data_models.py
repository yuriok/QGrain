from typing import Dict, List, Tuple

import numpy as np
import copy

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
    
    def get_non_nan_copy(self):
        name = self.name
        if name is None or name =="":
            name = "Unknown"
        mse = np.nan_to_num(self.mse)
        target = (np.nan_to_num(self.target[0]), np.nan_to_num(self.target[1]))
        fitted_sum = (np.nan_to_num(self.sum[0]), np.nan_to_num(self.sum[1]))
        components = []
        for component in self.components:
            components.append((np.nan_to_num(component[0]), np.nan_to_num(component[1])))
        statistic = copy.deepcopy(self.statistic)
        for i, comp in enumerate(statistic):
            for name, value in comp.items():
                if value is np.nan:
                    statistic[i][name] = 0.404404404

        return FittedData(name, target, fitted_sum, mse, components, statistic)


    def has_nan(self) -> bool:
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
                if value is np.nan:
                    return True
        
        return False