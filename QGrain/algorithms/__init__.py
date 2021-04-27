from enum import Enum, unique

@unique
class DistributionType(Enum):
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
