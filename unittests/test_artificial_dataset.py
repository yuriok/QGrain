import pytest
import numpy as np

from QGrain.models.artificial_dataset import *
from QGrain.statistics import *
from QGrain.generate import SIMPLE_PRESET, random_parameters

from scipy.stats import norm


class TestArtificialComponent:
    classes = np.logspace(0, 5, 101) * 0.02
    classes_phi = to_phi(classes)
    distribution = norm.pdf(classes_phi, 5, 1.0) * interval_phi(classes_phi)
    proportion = 0.5
    m, v, s, k = norm.stats(5, 1.0, moments="mvsk")
    std = np.sqrt(v)
    moments = (m, std, s, k)
    component = ArtificialComponent(classes, classes_phi, distribution, proportion, moments)

    def test_properties(self):
        assert self.component.mean == self.m
        assert self.component.sorting_coefficient == self.std
        assert self.component.skewness == self.s
        assert self.component.kurtosis == self.k
        assert isinstance(self.component.moments, dict)


class TestArtificialSample:
    classes = np.logspace(0, 5, 101) * 0.02
    classes_phi = to_phi(classes)
    c1 = norm.pdf(classes_phi, 10, 1.0) * interval_phi(classes_phi)
    c2 = norm.pdf(classes_phi, 7.5, 1.0) * interval_phi(classes_phi)
    c3 = norm.pdf(classes_phi, 5, 1.0) * interval_phi(classes_phi)
    distribution = c1 * 0.1 + c2 * 0.4 + c3 * 0.5
    components = [c1, c2, c3]
    proportions = [0.1, 0.4, 0.5]
    m, v, s, k = norm.stats([10, 7.5, 5], [1.0, 1.0, 1.0], moments="mvsk")
    std = np.sqrt(v)
    moments = (m, std, s, k)
    sample = ArtificialSample("Sample", classes, classes_phi, distribution, components, proportions, moments)

    def test_iter(self):
        assert len(self.sample) == 3
        for i, component in enumerate(self.sample):
            assert component.mean == [10, 7.5, 5][i]
            assert component.sorting_coefficient == 1.0

    def test_index(self):
        assert self.sample[0].mean == 10.0
        assert self.sample[-1].mean == 5.0

    def test_slice(self):
        for component in self.sample[:-1]:
            pass


class TestArtificialDataset:
    parameters = random_parameters(SIMPLE_PRESET["target"], 100)
    dataset = ArtificialDataset(parameters, SIMPLE_PRESET["distribution_type"])

    def test_get_item(self):
        sample = self.dataset[0]
        assert isinstance(sample, ArtificialSample)

    def test_reference(self):
        assert id(self.dataset[0].classes) == id(self.dataset.classes)
        assert id(self.dataset[0].classes_phi) == id(self.dataset.classes_phi)

    def test_iter(self):
        for i, sample in enumerate(self.dataset):
            assert isinstance(sample, ArtificialSample)

    def test_reverse_iter(self):
        for i, sample in enumerate(reversed(self.dataset)):
            assert isinstance(sample, ArtificialSample)

    def test_next(self):
        iterator = iter(self.dataset)
        i = 0
        try:
            while True:
                sample = next(iterator)
                assert isinstance(sample, ArtificialSample)
                i += 1
        except StopIteration:
            assert i == len(self.dataset)

    def test_slice(self):
        for sample in self.dataset[3:]:
            assert isinstance(sample, ArtificialSample)

    def test_consistent(self):
        for i, sample in enumerate(self.dataset):
            assert sample.distribution.tobytes() == self.dataset.distributions[i].tobytes()


if __name__ == "__main__":
    pytest.main(["-s"])
