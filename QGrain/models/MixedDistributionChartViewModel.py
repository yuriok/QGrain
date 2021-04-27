import numpy as np

class MixedDistributionChartViewModel:
    def __init__(self, classes_φ, target,
                 mixed, distributions, fractions,
                 component_prefix="C", title="", **kwargs):
        self.classes_φ = classes_φ
        self.mixed = mixed
        self.distributions = distributions
        self.fractions = fractions
        self.target=target
        self.component_prefix = component_prefix
        self.title = title
        self.kwargs = kwargs

    @property
    def n_components(self) -> int:
        return len(self.distributions)

def get_demo_view_model() -> MixedDistributionChartViewModel:
    from scipy.stats import norm
    classes_μm = np.logspace(0, 5, 101)*0.02
    classes_φ = -np.log2(classes_μm/1000.0)
    mixed = np.zeros_like(classes_φ)
    locs = [10.5, 7.5, 5.0]
    scales = [1.0, 1.0, 1.0]
    fractions = [0.2, 0.3, 0.5]
    distributions = []
    interval = abs((classes_φ[-1] - classes_φ[0]) / (len(classes_φ) - 1))
    for loc, scale, fraction in zip(locs, scales, fractions):
        distribution = norm.pdf(classes_φ, loc=loc, scale=scale) * interval
        distributions.append(distribution)
        mixed += distribution * fraction
    model = MixedDistributionChartViewModel(classes_φ, mixed,
                                        mixed, distributions, fractions,
                                        title="Demo")
    return model
