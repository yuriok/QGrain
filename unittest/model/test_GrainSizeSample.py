import numpy as np
import pytest
from QGrain.model import GrainSizeDataset, GrainSizeSample
from scipy.stats import norm


def get_interval_φ(classes_φ: np.ndarray)-> float:
    return abs((classes_φ[0]-classes_φ[-1]) / (len(classes_φ)-1))


class TestGrainSizeSample:
    def setup_class(self):
        self.sample_name = "Test"
        self.classes_μm = np.logspace(0, 5, 101) * 0.02
        self.classes_φ = -np.log2(self.classes_μm / 1000)
        self.interval_φ = get_interval_φ(self.classes_φ)
        self.distribution = norm.pdf(self.classes_φ, loc=5.0, scale=2.0) * self.interval_φ
        assert np.abs(np.sum(self.distribution)) - 1 < 1e-2

    def test_ctor(self):
        sample = GrainSizeSample(
            self.sample_name,
            self.classes_μm,
            self.classes_φ,
            self.distribution)

    # It must support these interfaces.
    def test_interface_accessible(self):
        sample = GrainSizeSample(
            self.sample_name,
            self.classes_μm,
            self.classes_φ,
            self.distribution)
        sample.name
        sample.uuid
        sample.classes_μm
        sample.classes_φ
        sample.distribution
