import pytest
from scipy.stats import norm

from QGrain.statistics import *

classes_phi = np.linspace(15, -15, 101)
classes = to_microns(classes_phi)
interval = interval_phi(classes_phi)
mean_values = np.random.random(100)*20-10 + 0.0
distributions = [norm.pdf(classes_phi, mean, 1.0) * interval for mean in mean_values]


def test_mode():
    for mean, distribution in zip(mean_values, distributions):
        mode_size = mode(classes, classes_phi, distribution, is_geometric=False)
        # the bias is less than the interval
        assert abs(mode_size - mean) < interval


def test_modes():
    mean_values_1 = np.random.randn(100) * 0.1 + 10.5
    mean_values_2 = np.random.randn(100) * 0.1 + 5.5
    proportions = np.random.randn(100) * 0.01 + 0.5
    mixed_distributions = [(proportions[i] * norm.pdf(classes_phi, mean_values_1[i], 1.0) +
                            (1 - proportions[i]) * norm.pdf(classes_phi, mean_values_2[i], 1.0)) *
                           interval for i in range(100)]
    for mean_1, mean_2, distribution in zip(mean_values_1, mean_values_2, mixed_distributions):
        mode_values, frequencies = modes(classes, classes_phi, distribution, is_geometric=False)
        # assert 0 < len(mode_values) < 3
        # the bias is less than the interval
        if len(mode_values) == 1:
            assert min(abs(mode_values[0] - mean_1), abs(mode_values[0] - mean_2)) < interval
        elif len(mode_values) == 2:
            assert abs(mode_values[0] - mean_1) < interval
            assert abs(mode_values[1] - mean_2) < interval


def test_to_cumulative():
    for (i, distribution) in enumerate(distributions):
        cumulative = to_cumulative(distribution)
        assert np.all(np.less_equal(cumulative[:-1], cumulative[1:]))


def test_ppf():
    medians = []
    for (i, distribution) in enumerate(distributions):
        ppf = reversed_phi_ppf(classes_phi, distribution)
        medians.append(ppf(0.5))
    medians = np.array(medians)
    assert np.mean(np.abs(medians - norm.median(mean_values, 1.0))) < interval


def test_interval_phi():
    mean_interval = abs(np.mean(classes_phi[:-1] - classes_phi[1:]))
    assert abs(interval - mean_interval) < 1e-8


def test_to_phi_microns():
    sizes = np.copy(classes)
    assert np.all(np.less(sizes - to_microns(to_phi(sizes)), 1e-8))


def test_arithmetic():
    # ONLY TEST THE EXISTING OF KEYS
    keys = ["mean", "std", "skewness", "kurtosis"]
    for (i, distribution) in enumerate(distributions):
        statistics = arithmetic(classes, distribution)
        for key in keys:
            assert key in statistics.keys()


def test_other_four_methods():
    # ONLY TEST THE EXISTING OF KEYS
    keys = ["mean", "std", "skewness", "kurtosis", "std_description", "skewness_description", "kurtosis_description"]
    for (i, distribution) in enumerate(distributions):
        statistics = geometric(classes, distribution)
        for key in keys:
            assert key in statistics.keys()
        statistics = logarithmic(classes_phi, distribution)
        for key in keys:
            assert key in statistics.keys()
        _ppf = reversed_phi_ppf(classes_phi, distribution)
        statistics = geometric_fw57(_ppf)
        for key in keys:
            assert key in statistics.keys()
        statistics = logarithmic_fw57(_ppf)
        for key in keys:
            assert key in statistics.keys()


def test_proportion_methods():
    for (i, distribution) in enumerate(distributions):
        proportions_gsm(classes_phi, distribution)
        proportions_ssc(classes_phi, distribution)
        proportions_bgssc(classes_phi, distribution)
        all_proportions(classes_phi, distribution)


def test_major_statistics():
    keys = ["mean", "std", "skewness", "kurtosis", "std_description", "skewness_description", "kurtosis_description",
            "mode", "modes", "median", "mean_description"]
    for (i, distribution) in enumerate(distributions):
        for j in range(4):
            statistics = major_statistics(classes, classes_phi, distribution, is_geometric=i % 2, is_fw57=i//2)
            for key in keys:
                assert key in statistics.keys()


def test_all_statistics():
    keys = ["arithmetic", "geometric", "logarithmic", "geometric_fw57", "logarithmic_fw57",
            "proportions_gsm", "proportions_ssc", "proportions_bgssc", "proportions", "group_folk54", "group_bp12"]
    for (i, distribution) in enumerate(distributions):
        statistics = all_statistics(classes, classes_phi, distribution)
        for key in keys:
            assert key in statistics.keys()


if __name__ == "__main__":
    pytest.main(["-s"])
