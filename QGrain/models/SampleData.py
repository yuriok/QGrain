
from enum import Enum, unique
from uuid import uuid4, UUID
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

@unique
class SampleTag(Enum):
    Default = 1
    Ignored = 2


class SampleData:
    """This class is used to store all the necessary data of each sample which can be fitted.

    All its attributes have been packed with read-only properties to avoid the modification by mistake.

    Note: Do not call `ctor` to create instances because it won't check the validity of input arguments.
    You can use the `add_batch` func of class `SampleDataset` instead.

    Attributes:
        uuid: A `uuid.UUID` object.
        name: A human readable `str`.
        classes: A `numpy.ndarray` to represent the grain size classes.
        distribution: A `numpy.ndarray` to represent the distribution data of this sample.
        tag: A `SampleTag` enum to indicate that how to handle this sample. Use `reset_tag`, `ignore` to control the tag.
    """
    def __init__(self, name: str, classes: np.ndarray, distribution: np.ndarray):
        self.__uuid = uuid4()
        self.__name = name
        self.__classes = classes
        self.__distribution = distribution
        self.__tag = SampleTag.Default

    @property
    def uuid(self) -> UUID:
        return self.__uuid
    
    @property
    def name(self) -> str:
        return self.__name

    @property
    def classes(self) -> np.ndarray:
        return self.__classes

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution
    
    @property
    def tag(self) -> SampleTag:
        return self.__tag

    def reset_tag(self):
        self.__tag = SampleTag.Default

    def ignore(self):
        self.__tag = SampleTag.Ignored
