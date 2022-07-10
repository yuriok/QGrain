from __future__ import annotations

import numpy as np
from typing import *
from ..statistics import interval_phi


def _incremental(classes: Sequence[int | float]) -> Tuple[bool, int | None]:
    """
    Check if the series of grain size classes is incremental.

    ## Parameters

    classes: The grain size classes.

    ## Returns

    is_incremental: If the series of classes is incremental.

    error_index: If it is incremental, return `None`, else return the index of first invalid value.
    """
    classes = tuple(classes)
    for i, (left, right) in enumerate(zip(classes[:-1], classes[1:])):
        if left >= right:
            return False, i
    else:
        return True, None


def _error_text(array: np.ndarray, index: int):
    w = 3
    l, r = max(index - w, 0), min(index + w, len(array))
    l_header = [] if l == 0 else ["..."]
    r_header = [] if r == len(array) else ["..."]
    indicator = [""] * len(array)
    indicator[index] = "↑"
    indicator[min(index + 1, len(array) - 1)] = "↑"
    error_text = "\n".join(
        ["\t".join(["Index   "] + l_header + [f"{i:<10}" for i in range(len(array))[l:r]] + r_header),
         "\t".join(["Value   "] + l_header + [f"{f'{v:.4f}':<10}" for v in array[l:r]] + r_header),
         "\t".join(["Position"] + l_header + [f"{s:<10}" for s in indicator[l:r]] + r_header)])
    return error_text


def validate_classes(classes: Sequence[float]) -> Tuple[bool, np.ndarray | str]:
    """
    Check if the series of grain size classes is valid.

    ## Parameters

    classes: The grain size classes.

    ## Returns

    is_valid: If the series of classes is valid.

    array_or_msg: If it is valid, return a new numpy array, else return the error message.
    """
    if classes is None:
        return False, "The passed `classes` can not be `None`."
    array: np.ndarray = np.array(classes, dtype=np.float32)
    if array.ndim != 1:
        return False, "The passed `classes` should be one-dimensional."
    if len(array) == 0:
        return False, "The passed `classes` can not be empty."
    indices = np.arange(len(array))
    nan_indices: np.ndarray = indices[np.isnan(array)]
    if len(nan_indices) > 0:
        return False, (f"There is at least one NaN value in the series of grain size classes. "
                       f"Check the index(es): {', '.join(nan_indices.astype(str))}.")
    incremental, index = _incremental(array)
    if not incremental:
        error_text = _error_text(array, index)
        return False, f"The series of grain size classes is not incremental.\n{error_text}"
    classes_phi = -np.log2(array / 1000)
    mean_interval = interval_phi(classes_phi)
    intervals = classes_phi[:-1] - classes_phi[1:]
    absolute_errors = np.abs(intervals - mean_interval)
    index = np.argmax(absolute_errors)
    if absolute_errors[index] > 0.05:
        error_text = _error_text(array, index)
        return False, (f"The grain size classes are not evenly spaced on a log scale. "
                       f"The max absolute error of intervals is {absolute_errors[index]} phi.\n{error_text}")
    return True, array


def validate_distributions(distributions: Sequence[Sequence[float]]) -> Tuple[bool, np.ndarray | str]:
    if distributions is None:
        return False, "The passed `distributions` can not be `None`."
    array = np.array(distributions, dtype=np.float32)
    if array.ndim != 2:
        return False, "The passed `distributions` should be two-dimensional."
    n_samples, n_classes = array.shape
    if n_samples == 0 or n_classes == 0:
        return False, "The passed `distribution` can not be empty."
    cols, rows = np.meshgrid(np.arange(n_classes), np.arange(n_samples))
    nan_keys = np.isnan(array)
    cells = [f"({row}, {col})" for row, col in zip(rows[nan_keys], cols[nan_keys])]
    if len(cells) > 0:
        return False, (f"There is at least one NaN value in the passed `distributions`. "
                       f"See the cell(s): {', '.join(cells)}")
    rows = np.arange(n_samples)
    summed = np.sum(array, axis=1)
    valid = (np.greater(summed, 0.95) & np.less(summed, 1.05)) | (np.greater(summed, 95) & np.less(summed, 105))
    invalid_rows = rows[np.logical_not(valid)]
    if len(invalid_rows) > 0:
        return False, (f"There is at least one distribution which has the sum not equal to 1 or 100. "
                       f"Check the following row(s): {', '.join(invalid_rows.astype(str))}.")
    array = array / np.sum(array, axis=1, keepdims=True)
    return True, array


class Sample:
    __slots__ = ("_name", "_classes", "_classes_phi", "_distribution")

    def __init__(self, name: str,
                 classes: np.ndarray,
                 classes_phi: np.ndarray,
                 distribution: np.ndarray):
        self._name = name
        self._classes = classes
        self._classes_phi = classes_phi
        self._distribution = distribution

    def __repr__(self):
        return f"Sample({self._name})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def classes(self) -> np.ndarray:
        return self._classes

    @property
    def classes_phi(self) -> np.ndarray:
        return self._classes_phi

    @property
    def interval_phi(self) -> float:
        return interval_phi(self.classes_phi)

    @property
    def distribution(self) -> np.ndarray:
        return self._distribution


class Dataset:
    def __init__(self, name: str, sample_names: Sequence[str],
                 classes: Sequence[int | float], distributions: Sequence[Sequence[int | float]]):
        assert isinstance(name, str)
        if len(name) == 0:
            raise ValueError("The name of dataset can not be empty.")
        for sample_name in sample_names:
            assert isinstance(sample_name, str)
            if len(sample_name) == 0:
                raise ValueError("The name of sample can not be empty.")
        valid, array_or_msg = validate_classes(classes)
        if not valid:
            raise ValueError(array_or_msg)
        self._classes = array_or_msg
        valid, array_or_msg = validate_distributions(distributions)
        if not valid:
            raise ValueError(array_or_msg)
        self._distributions = array_or_msg
        self._name = name
        self._classes_phi = -np.log2(self._classes / 1000)
        self._sample_names = list(sample_names)

    def __repr__(self) -> str:
        return f"Dataset({self._name})"

    def __len__(self) -> int:
        return len(self._sample_names)

    def __iter__(self):
        for i in range(len(self._sample_names)):
            yield self._get_sample(i)

    def __getitem__(self, key) -> Union[Sample, Sequence[Sample]]:
        if isinstance(key, int):
            return self._get_sample(key)
        elif isinstance(key, slice):
            return [self._get_sample(index) for index in np.arange(len(self._sample_names))[key]]
        else:
            raise TypeError(f"Sample indices must be integers or slices, not {type(key)}.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def sample_names(self) -> List[str]:
        return self._sample_names.copy()

    @property
    def samples(self) -> List[Sample]:
        return list(iter(self))

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    @property
    def classes(self) -> np.ndarray:
        return self._classes

    @property
    def classes_phi(self) -> np.ndarray:
        return self._classes_phi

    @property
    def interval_phi(self) -> float:
        return interval_phi(self.classes_phi)

    @property
    def distributions(self) -> np.ndarray:
        return self._distributions

    def _get_sample(self, index: int) -> Sample:
        name = self._sample_names[index]
        distribution = self._distributions[index]
        sample = Sample(name, self._classes, self._classes_phi, distribution)
        return sample
