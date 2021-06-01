QGRAIN_VERSION = "0.3.4"

import os

QGRAIN_ROOT_PATH = os.path.dirname(__file__)

from enum import Enum, unique

@unique
class DistributionType(Enum):
    Customized = "Customized"
    Nonparametric = "NonParametric"
    Normal = "Normal"
    Weibull = "Weibull"
    SkewNormal = "SkewNormal"

@unique
class FittingState(Enum):
    NotStarted = 0
    Fitting = 1
    Failed = 2
    Succeeded = 4
