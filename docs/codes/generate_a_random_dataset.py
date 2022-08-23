from QGrain.models import DistributionType
from QGrain.generate import random_dataset

preset = dict(target=[
    [(0.0, 0.0), (10.2, 0.0), (1.1, 0.0), (1.0, 0.1)],
    [(0.0, 0.0), (7.5, 0.0), (1.2, 0.0), (2.0, 0.2)],
    [(0.0, 0.0), (5.0, 0.0), (1.0, 0.0), (2.5, 0.5)]],
    distribution_type=DistributionType.SkewNormal)

dataset = random_dataset(**preset, n_samples=100,
                         min_size=0.02, max_size=2000.0, n_classes=101,
                         precision=4, noise=5)
