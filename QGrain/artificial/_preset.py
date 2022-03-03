from ..ssu import DistributionType


SIMPLE_PRESET = dict(
    target=[
        [(0.0, 0.0), (10.2, 0.0), (1.1, 0.0), (1.0, 0.1)],
        [(0.0, 0.0), (7.5, 0.0), (1.2, 0.0), (2.0, 0.2)],
        [(0.0, 0.0), (5.0, 0.0), (1.0, 0.0), (2.5, 0.5)]],
    distribution_type=DistributionType.SkewNormal)

LOESS_PRESET = dict(
    target=[
        [(0.0, 0.10), (10.2, 0.1), (1.1, 0.1), (1.0, 0.1)],
        [(0.0, 0.10), (7.5, 0.1), (1.2, 0.1), (2.0, 0.1)],
        [(0.0, 0.10), (5.0, 0.2), (1.0, 0.1), (2.5, 0.2)]],
    distribution_type=DistributionType.SkewNormal)

LACUSTRINE_PRESET = dict(
    target=[
        [(0.0, 0.10), (10.2, 0.1), (1.1, 0.1), (1.0, 0.1)],
        [(0.0, 0.10), (7.5, 0.1), (1.2, 0.1), (2.0, 0.1)],
        [(0.0, 0.10), (5.0, 0.2), (1.0, 0.1), (2.5, 0.2)],
        [(0.0, 0.10), (2.5, 0.4), (1.0, 0.1), (1.0, 0.2)]],
    distribution_type=DistributionType.SkewNormal)
