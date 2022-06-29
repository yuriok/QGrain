from __future__ import annotations

import typing
from uuid import UUID, uuid4

import numpy as np

from ..statistics import get_interval_φ


class ClassesNotIncrementalError(Exception):
    """Raises while the array of grain size classes is not incremental.

    It's just an ASSUMPTION for the convenience of coding.
    """
    pass


class ClassesNotMatchError(Exception):
    """Raises while the grain size classes of a new batch of sample data are not equal to the existing classes."""
    pass


class NaNError(Exception):
    """Raises while there is at least one NaN value in the array."""
    pass


class ArrayEmptyError(Exception):
    """Raises while the length of array is 0."""
    pass


class SampleNameEmptyError(Exception):
    """Raises while the name of sample is empty."""
    pass


class DistributionSumError(Exception):
    """Raises while the sum of distribution values is not equal to 1 or 100."""
    pass


class SampleNotFountError(Exception):
    """Raises while the sample is not found in this dataset."""
    pass


class GrainSizeSample:
    def __init__(self, name: str,
                 classes_μm: np.ndarray,
                 classes_φ: np.ndarray,
                 distribution: np.ndarray):
        self.__uuid = uuid4()
        self.__name = name
        self.__classes_μm = classes_μm
        self.__classes_φ = classes_φ
        self.__distribution = distribution

    @property
    def uuid(self) -> UUID:
        return self.__uuid

    @property
    def name(self) -> str:
        return self.__name

    @property
    def classes_μm(self) -> np.ndarray:
        return self.__classes_μm

    @property
    def classes_φ(self) -> np.ndarray:
        return self.__classes_φ

    @property
    def interval_φ(self) -> float:
        return get_interval_φ(self.classes_φ)

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution


class GrainSizeDataset:
    def __init__(self):
        self.__classes_μm = None # type: np.ndarray
        self.__classes_φ = None # type: np.ndarray
        self.__id_table = {} # type: dict[UUID, GrainSizeSample]
        self.__samples = [] # type: list[GrainSizeSample]

    @property
    def n_samples(self) -> int:
        return len(self.__samples)

    @property
    def n_classes(self) -> int:
        if self.__classes_μm is None:
            return 0
        else:
            return len(self.__classes_μm)

    @property
    def has_sample(self) -> bool:
        if self.n_samples == 0:
            return False
        else:
            return True

    @property
    def classes_μm(self) -> np.ndarray:
        return self.__classes_μm

    @property
    def classes_φ(self) -> np.ndarray:
        return self.__classes_φ

    @property
    def interval_φ(self) -> float:
        return get_interval_φ(self.classes_φ)

    @property
    def samples(self) -> typing.List[GrainSizeSample]:
        return self.__samples.copy()

    @property
    def distribution_matrix(self) -> np.ndarray:
        distributions = [sample.distribution for sample in self.samples]
        matrix = np.array(distributions)
        return matrix

    @staticmethod
    def is_incremental(nums: np.ndarray) -> bool:
        """Returns `True` while the array is incremental.
        This method is used to validate the array of grain size classes."""
        # for i in range(1, len(nums)):
        #     if nums[i] <= nums[i-1]:
        #         return False
        # return True
        return np.all(np.less(nums[:-1], nums[1:]))

    @staticmethod
    def validate_classes_μm(classes: np.ndarray):
        assert classes is not None
        assert type(classes) == np.ndarray
        assert classes.dtype == np.float64
        if len(classes) == 0:
            raise ArrayEmptyError("The array of grain size classes is empty.")
        if np.any(np.isnan(classes)):
            raise NaNError("There is at least one NaN value in grain size classes.")
        # It may raises when the users select a wrong data file.
        if not GrainSizeDataset.is_incremental(classes):
            raise ClassesNotIncrementalError("The array of grain size classes is not incremental.")

    @staticmethod
    def validate_sample_name(sample_name: str):
        assert sample_name is not None
        assert type(sample_name) == str
        if sample_name == "":
            raise SampleNameEmptyError("Sample name is empty.")

    @staticmethod
    def validate_distribution(distribution: np.ndarray):
        assert distribution is not None
        assert type(distribution) == np.ndarray
        assert distribution.dtype == np.float64
        if len(distribution) == 0:
            raise ArrayEmptyError("The array of distribution of this sample is empty.")
        if np.any(np.isnan(distribution)):
            raise NaNError("There is at least one NaN value in the distribution.")
        # Check if the sum is close to 1 or 100 (some users may use percentage).
        s = np.sum(distribution)
        if (s > 0.95 and s < 1.05):
            pass
        elif (s > 95 and s < 105):
            # If the sum is close to 100, make it close to 1.
            np.true_divide(distribution, 100.0, out=distribution)
        else:
            raise DistributionSumError("Distribution of this sample not sum equal to 1 or 100.")

    @staticmethod
    def are_classes_match(classes_a: np.ndarray, classes_b: np.ndarray):
        if len(classes_a) != len(classes_b):
            return False
        equal_res = np.equal(classes_a, classes_b)
        if not np.all(equal_res):
            return False
        return True

    def add_batch(self, classes_μm: np.ndarray,
                  names: typing.List[str],
                  distributions: typing.List[np.ndarray],
                  need_validation: bool = True):
        self.validate_classes_μm(classes_μm)
        # If there is any sample already, it's necessary to check the consistency of classes.
        if self.has_sample:
            if not self.are_classes_match(self.classes_μm, classes_μm):
                raise ClassesNotMatchError(self.classes_μm, classes_μm)
        assert len(names) == len(distributions)

        self.__classes_μm = classes_μm
        self.__classes_φ = -np.log2(classes_μm/1000.0)
        # use temp to implement roll-back
        temp_table = {}
        temp_list = []
        for name, distribution in zip(names, distributions):
            assert len(distribution) == len(classes_μm)
            if need_validation:
                self.validate_sample_name(name)
                self.validate_distribution(distribution)
            sample = GrainSizeSample(name, self.classes_μm, self.classes_φ, distribution)
            temp_table.update({sample.uuid: sample})
            temp_list.append(sample)
        # if no exception raised, the codes below will be executed
        self.__id_table.update(temp_table)
        self.__samples.extend(temp_list)

    def combine(self, another_dataset: GrainSizeDataset):
        if self.are_classes_match(self.classes_μm, another_dataset.classes_μm):
            self.__id_table.update(another_dataset.__id_table)
            self.__samples.extend(another_dataset.__samples)
        else:
            raise ClassesNotMatchError(self.classes_μm, another_dataset.classes_μm)

    def get_sample_by_id(self, uuid: UUID):
        if uuid in self.__id_table:
            return self.__id_table[uuid]
        else:
            raise SampleNotFountError("There is no sample with this id.", uuid)
