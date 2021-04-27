__all__ = ["FittingTask"]

from uuid import uuid4

from QGrain.algorithms import DistributionType
from QGrain.models.GrainSizeSample import GrainSizeSample


class FittingTask:
    def __init__(self, sample: GrainSizeSample,
                 distribution_type: DistributionType,
                 n_components: int,
                 resolver="classic",
                 resolver_setting=None,
                 initial_guess=None,
                 reference=None):
        self.uuid = uuid4()
        self.sample = sample
        self.distribution_type = distribution_type
        self.n_components = n_components
        self.resolver = resolver
        self.resolver_setting = resolver_setting
        self.initial_guess = initial_guess
        self.reference = reference
