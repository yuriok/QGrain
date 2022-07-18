from enum import Enum, unique


@unique
class DistributionType(Enum):
    Normal = 1
    SkewNormal = 2
    Weibull = 3
    GeneralWeibull = 4


@unique
class KernelType(Enum):
    Nonparametric = 0
    Normal = 1
    SkewNormal = 2
    Weibull = 3
    GeneralWeibull = 4


from .dataset import *
from .artificial_dataset import *
from .ssu_result import *
from .emma_result import *
from .udm_result import *
