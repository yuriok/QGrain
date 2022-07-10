from uuid import uuid4

from ..models import Sample
from ._distribution import DistributionType


class SSUTask:
    def __init__(
            self, sample: Sample,
            distribution_type: DistributionType,
            n_components: int,
            resolver_setting=None,
            initial_parameters=None):
        self.uuid = uuid4()
        self.sample = sample
        self.distribution_type = distribution_type
        self.n_components = n_components
        self.resolver_setting = resolver_setting
        self.initial_parameters = initial_parameters
