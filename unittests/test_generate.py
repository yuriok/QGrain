import numpy as np
import pytest

from QGrain.distributions import get_distribution
from QGrain.generate import *


def test_presets():
    random_dataset(**SIMPLE_PRESET, n_samples=100)
    random_dataset(**LOESS_PRESET, n_samples=100)
    random_dataset(**LACUSTRINE_PRESET, n_samples=100)


def test_random_parameters():
    n_samples = 1000000
    parameters = random_parameters(SIMPLE_PRESET["target"], n_samples=n_samples)
    distribution_class = get_distribution(SIMPLE_PRESET["distribution_type"])
    assert parameters.ndim == 3
    assert parameters.shape == (n_samples, distribution_class.N_PARAMETERS + 1, len(SIMPLE_PRESET["target"]))
    for i, component in enumerate(SIMPLE_PRESET["target"]):
        for j, (mean, std) in enumerate(component):
            assert abs(np.mean(parameters[:, j, i]) - mean) < 5e-3
            assert abs(np.std(parameters[:, j, i]) - std) < 5e-3


def test_random_dataset():
    n_samples = 100
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=n_samples)
    assert len(dataset) == n_samples
    assert dataset.n_components == len(SIMPLE_PRESET["target"])


def test_random_sample():
    sample = random_sample(**SIMPLE_PRESET)
    assert len(sample) == len(SIMPLE_PRESET["target"])


def test_random_mean_sample():
    from scipy.special import softmax
    mean_sample = random_mean_sample(**SIMPLE_PRESET)
    proportions = softmax(np.array([SIMPLE_PRESET["target"][i][3][0] for i in range(3)]))
    for i, component in enumerate(SIMPLE_PRESET["target"]):
        assert mean_sample[i].mean == component[1][0]
        assert mean_sample[i].sorting_coefficient == component[2][0]
        assert mean_sample[i].proportion == proportions[i]


if __name__ == "__main__":
    pytest.main(["-s"])
