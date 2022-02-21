__all__ = ["Reference"]

import numpy as np

from ._distribution import DISTRIBUTION_CLASS_MAP, DistributionType


class Reference:
    def __init__(self,
                 distribution_type: DistributionType,
                 params: np.ndarray):
        assert isinstance(distribution_type, DistributionType)
        assert isinstance(params, np.ndarray)
        assert params.ndim == 2
        n_params, n_components = params.shape
        assert n_params == DISTRIBUTION_CLASS_MAP[distribution_type].N_PARAMS
        self.__distribution_type = distribution_type
        self.__params = params
        self.__n_components = n_components
        self.__n_params = n_params

    @property
    def distribution_type(self) -> DistributionType:
        return self.__distribution_type

    @property
    def n_components(self) -> int:
        return self.__n_components

    @property
    def n_params(self) -> int:
        return self.__n_params

    @property
    def params(self) -> np.ndarray:
        return self.__params
