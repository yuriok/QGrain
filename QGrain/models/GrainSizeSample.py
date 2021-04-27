__all__ = ["GrainSizeSample"]

from uuid import UUID, uuid4

import numpy as np


class GrainSizeSample:
    def __init__(self, name: str, classes_μm: np.ndarray, classes_φ: np.ndarray, distribution: np.ndarray):
        self.__uuid = uuid4()
        self.__name = name
        self.__classes_μm = classes_μm
        self.__classes_φ  = classes_φ
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
    def interval_φ(self)-> float:
        return abs((self.__classes_φ[0]-self.__classes_φ[-1]) / (len(self.__classes_φ)-1))

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution
