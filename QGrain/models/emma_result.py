__all__ = ["EMMAResult"]

import copy
from typing import *

import numpy as np
from numpy import ndarray

from ..models import KernelType, Dataset, ArtificialDataset
from ..metrics import loss_numpy


class EMMAResult:
    """The class to represent the result of EMMA algorithm."""
    def __init__(self, dataset: [ArtificialDataset, Dataset], kernel_type: KernelType, n_members: int,
                 proportions: ndarray, end_members: ndarray, time_spent: Union[int, float], x0: ndarray = None,
                 settings: Dict[str, Any] = None, loss_series: Dict[str, ndarray] = None):
        # do some validations
        assert isinstance(dataset, (ArtificialDataset, Dataset))
        assert isinstance(kernel_type, KernelType)
        assert isinstance(n_members, int)
        assert isinstance(proportions, ndarray)
        n_iterations = len(proportions)
        assert proportions.ndim == 3
        assert proportions.shape == (n_iterations, len(dataset), n_members)
        assert isinstance(end_members, ndarray)
        assert end_members.ndim == 3
        assert end_members.shape == (n_iterations, n_members, len(dataset.classes))
        assert isinstance(time_spent, (int, float))
        if x0 is not None:
            assert isinstance(x0, ndarray)
            assert x0.ndim == 2
            assert x0.shape[1] == n_members
        if settings is not None:
            assert isinstance(settings, dict)
            if loss_series is not None:
                assert settings["loss"] in loss_series
                if settings["need_history"]:
                    assert len(loss_series[settings["loss"]]) == proportions.shape[0]

        self._dataset = dataset
        self._kernel_type = kernel_type
        self._n_members = n_members
        self._x0 = x0
        self._proportions = proportions
        self._end_members = end_members
        self._i = -1
        self._time_spent = time_spent
        self._settings = settings
        self._loss_series = loss_series
        modes = [(i, dataset.classes[np.unravel_index(np.argmax(end_members[-1, i]), end_members[-1, i].shape)])
                 for i in range(n_members)]
        modes.sort(key=lambda x: x[1])
        self._sorted_indexes = tuple([i for i, _ in modes])
        self._sort()

    @property
    def dataset(self) -> Union[ArtificialDataset, Dataset]:
        return self._dataset

    @property
    def n_samples(self) -> int:
        return len(self._dataset)

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
    def proportions(self) -> ndarray:
        return self._proportions[self._i]

    @property
    def end_members(self) -> ndarray:
        return self._end_members[self._i]

    @property
    def distributions(self) -> ndarray:
        return self.proportions @ self.end_members

    @property
    def time_spent(self) -> Union[int, float]:
        return self._time_spent

    @property
    def x0(self) -> Optional[ndarray]:
        return self._x0

    @property
    def n_iterations(self) -> int:
        return self._proportions.shape[0]

    @property
    def history(self):
        for i in range(self._proportions.shape[0]):
            copy_result = copy.copy(self)
            copy_result._i = i
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
            proportions[:, :, i] = self._proportions[:, :, j]
            end_members[:, i, :] = self._end_members[:, j, :]
        self._proportions = proportions
        self._end_members = end_members

    def loss(self, name: str) -> float:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = self.proportions @ self.end_members
        return loss_func(prediction, observation, None)

    def loss_series(self, name: str) -> ndarray:
        if name in self._loss_series:
            return self._loss_series[name]
        loss_func = loss_numpy(name)
        try:
            observation = np.expand_dims(self._dataset.distributions, 0).repeat(self.n_iterations, axis=0)
            prediction = self._proportions @ self._end_members
            series = loss_func(prediction, observation, (1, 2))
            self._loss_series[name] = series
            return series
        except MemoryError:
            series = []
            for result in self.history:
                series.append(result.loss(name))
            series = np.array(series)
            self._loss_series[name] = series
            return series

    def class_wise_losses(self, name: str) -> ndarray:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = self.proportions @ self.end_members
        losses = loss_func(prediction, observation, 0)
        return losses

    def sample_wise_losses(self, name: str) -> ndarray:
        loss_func = loss_numpy(name)
        observation = self._dataset.distributions
        prediction = self.proportions @ self.end_members
        losses = loss_func(prediction, observation, 1)
        return losses
