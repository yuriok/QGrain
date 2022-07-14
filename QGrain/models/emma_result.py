

import copy
from typing import *

import numpy as np
from numpy import ndarray

from ..models import Dataset
from ..metrics import loss_numpy
from ..kernels import KernelType


class EMMAResult:
    def __init__(self, dataset: Dataset, kernel_type: KernelType, n_members: int,
                 proportions: ndarray, end_members: ndarray, time_spent: float,
                 x0: ndarray = None, history: Sequence[Tuple[ndarray, ndarray]] = None,
                 settings: Dict[str, Any] = None, loss_series: Dict[str, ndarray] = None):
        self._dataset = dataset
        self._kernel_type = kernel_type
        self._n_members = n_members
        self._x0 = x0
        self._proportions = proportions
        self._end_members = end_members
        self._time_spent = time_spent
        self._history = history
        self._settings = settings
        self._loss_series = loss_series
        modes = [(i, dataset.classes[np.unravel_index(np.argmax(end_members[i]), end_members[i].shape)])
                 for i in range(n_members)]
        modes.sort(key=lambda x: x[1])
        self._sorted_indexes = tuple([i for i, _ in modes])
        self._sort()

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    @property
    def n_members(self) -> int:
        return self._n_members

    @property
    def n_classes(self) -> int:
        return len(self._dataset.classes_phi)

    @property
    def kernel_type(self) -> KernelType:
        return self._kernel_type

    @property
    def proportions(self) -> np.ndarray:
        return self._proportions

    @property
    def end_members(self) -> np.ndarray:
        return self._end_members

    @property
    def time_spent(self) -> Union[int, float]:
        return self._time_spent

    @property
    def x0(self) -> Optional[ndarray]:
        return self._x0

    @property
    def n_iterations(self) -> int:
        if self._history is None:
            return 0
        else:
            return len(self._history)

    @property
    def history(self):
        if self.n_iterations == 0:
            raise ValueError("No history record.")
        else:
            for fractions, end_members in self._history:
                copy_result = copy.copy(self)
                copy_result._proportions = fractions
                copy_result._end_members = end_members
                copy_result._sort()
                yield copy_result

    @property
    def settings(self) -> Optional[Dict[str, Any]]:
        if self._settings is None:
            return None
        else:
            return self._settings.copy()

    def _sort(self):
        proportions = np.zeros_like(self._proportions)
        end_members = np.zeros_like(self._end_members)
        for i, j in enumerate(self._sorted_indexes):
            proportions[:, i] = self._proportions[:, j]
            end_members[i, :] = self._end_members[j, :]
        self._proportions = proportions
        self._end_members = end_members

    def loss(self, name: str) -> float:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = self._proportions @ self._end_members
        return loss_func(prediction, observation, None)

    def loss_series(self, name: str) -> ndarray:
        if self.n_iterations == 0:
            raise ValueError("No history record.")
        else:
            series = []
            loss_func = loss_numpy(name)
            for proportions, end_members in self._history:
                predict = proportions @ end_members
                distance = loss_func(predict, self._dataset.distributions, None)
                series.append(distance)
            series = np.array(series)
            return series

    def class_wise_losses(self, name: str) -> ndarray:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = self._proportions @ self._end_members
        losses = loss_func(prediction, observation, 0)
        return losses

    def sample_wise_losses(self, name: str) -> ndarray:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = self._proportions @ self._end_members
        losses = loss_func(prediction, observation, 1)
        return losses
