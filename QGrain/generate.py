__all__ = ["SIMPLE_PRESET", "LOESS_PRESET", "LACUSTRINE_PRESET",
           "random_parameters", "random_dataset", "random_sample", "random_mean_sample"]

from typing import *

import numpy as np

from .models import DistributionType, ArtificialDataset

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


def random_parameters(target: Sequence[Sequence[Tuple[float, float]]], n_samples: int):
    """
    Get an array of random parameters which can be used to generate the artificial dataset.

    :param target: It defines the mean and standard deviation of each parameter of each component.
    :param n_samples: The number of samples to generate.
    :return: A numpy array (n_samples, n_parameters, n_components) of the generated parameters.
    """
    n_components = len(target)
    n_parameters = len(target[0])
    parameters = np.random.randn(n_samples, n_parameters, n_components)
    for component_i, sub_target in enumerate(target):
        for param_i, (mean, std) in enumerate(sub_target):
            parameters[:, param_i, component_i] = parameters[:, param_i, component_i] * std + mean
    return parameters


def random_dataset(target: Sequence[Sequence[Tuple[float, float]]], distribution_type: DistributionType, n_samples: int,
                   min_size=0.02, max_size=2000.0, n_classes=101, precision=4, noise=5):
    """
    Generate a random dataset.

    For example, a 3-normal-components `target` is as follows.

    --------------------------- 3 Normal ----------------------------
    |       Location     Scale        Weight                        |
    |       Mean   S.D.  Mean   S.D.  Mean   S.D.                   |
    |    [[(10.2,  0.0), (1.1,  0.0), (1.0,  0.1)],   # Component 1 |
    |     [( 7.5,  0.0), (1.2,  0.0), (2.0,  0.2)],   # Component 2 |
    |     [( 5.0,  0.0), (1.0,  0.0), (2.5,  0.5)]],  # Component 3 |
    -----------------------------------------------------------------

    :param target: It defines the mean and standard deviation of each parameter of each component.
    :param distribution_type: The type of elementary distribution.
    :param n_samples: The number of samples to generate.
    :param min_size: The minimum grain size class in microns.
    :param max_size: The maximum grain size class in microns.
    :param n_classes: The number of grain size classes. More classes can make the distributions more detailed.
    :param precision: The precision level (decimals) of generated grain size distributions.
    :param noise: The standard deviation of the random noise matrix. Usually, should use `precision+1`.
    :return: An instance of the `ArtificialDataset` class.
    """
    parameters = random_parameters(target, n_samples)
    dataset = ArtificialDataset(
        parameters, distribution_type,
        min_size=min_size, max_size=max_size, n_classes=n_classes,
        precision=precision, noise=noise)
    return dataset


def random_sample(target: Sequence[Sequence[Tuple[float, float]]], distribution_type: DistributionType,
                  min_size=0.02, max_size=2000.0, n_classes=101, precision=4, noise=5):
    """
    Generate a random sample.

    :param target: It defines the mean and standard deviation of each parameter of each component.
    :param distribution_type: The type of elementary distribution.
    :param min_size: The minimum grain size class in microns.
    :param max_size: The maximum grain size class in microns.
    :param n_classes: The number of grain size classes. More classes can make the distributions more detailed.
    :param precision: The precision level (decimals) of generated grain size distributions.
    :param noise: The standard deviation of the random noise matrix. Usually, should use `precision+1`.
    :return: An instance of the `ArtificialSample` class.
    """
    parameters = random_parameters(target, 1)
    dataset = ArtificialDataset(
        parameters, distribution_type,
        min_size=min_size, max_size=max_size, n_classes=n_classes,
        precision=precision, noise=noise)
    return dataset[0]


def random_mean_sample(target: Sequence[Sequence[Tuple[float, float]]], distribution_type: DistributionType,
                       min_size=0.02, max_size=2000.0, n_classes=101, precision=4, noise=5):
    """
    Generate a sample using the `mean` values in `target` as the parameters.

    :param target: It defines the mean and standard deviation of each parameter of each component.
    :param distribution_type: The type of elementary distribution.
    :param min_size: The minimum grain size class in microns.
    :param max_size: The maximum grain size class in microns.
    :param n_classes: The number of grain size classes. More classes can make the distributions more detailed.
    :param precision: The precision level (decimals) of generated grain size distributions.
    :param noise: The standard deviation of the random noise matrix. Usually, should use `precision+1`.
    :return: An instance of the `ArtificialSample` class.
    """
    parameters = np.expand_dims(np.array([[mean for (mean, std) in component] for component in target]).T, 0)
    dataset = ArtificialDataset(
        parameters, distribution_type,
        min_size=min_size, max_size=max_size, n_classes=n_classes,
        precision=precision, noise=noise)
    return dataset[0]
