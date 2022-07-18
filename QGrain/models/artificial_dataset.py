__all__ = ["ArtificialComponent", "ArtificialSample", "ArtificialDataset"]

from typing import *

import numpy as np
from numpy import ndarray

from ..statistics import to_phi, to_microns, interval_phi
from ..models import DistributionType, Dataset, Sample
from ..distributions import get_distribution


class ArtificialComponent:
    """The class to represent one component of the artificial sample."""
    __slots__ = ("_classes", "_classes_phi", "_distribution", "_proportion", "_moments")

    def __init__(self, classes: ndarray, classes_phi: ndarray,
                 distribution: ndarray, proportion: float,
                 moments: Tuple[float, float, float, float]):
        """
        Construct an instance of the `ArtificialComponent` class.

        **If not necessary, do not manually create the sample, because it will not validate the passed parameters.**

        :param classes: The grain size classes in microns.
        :param classes_phi: The grain size classes in phi values.
        :param distribution: The frequency distribution of grain size classes.
            Note, the sum of frequencies should be equal to 1.
        :param moments: A tuple that contains the mean, standard deviation, skewness, and kurtosis, respectively.
        :return: An instance of the `ArtificialComponent` class.
        """
        self._classes = classes
        self._classes_phi = classes_phi
        self._distribution = distribution
        self._proportion = proportion
        m, std, s, k = moments
        self._moments = dict(mean=m, std=std, skewness=s, kurtosis=k)

    def __repr__(self):
        return f"AC({self._moments['mean']:.2f} phi, {self._proportion:.2%})"

    @property
    def classes(self) -> ndarray:
        return self._classes

    @property
    def classes_phi(self) -> ndarray:
        return self._classes_phi

    @property
    def interval_phi(self) -> float:
        return interval_phi(self.classes_phi)

    @property
    def distribution(self) -> ndarray:
        return self._distribution

    @property
    def proportion(self) -> float:
        return self._proportion

    @property
    def moments(self) -> Dict[str, float]:
        return self._moments

    @property
    def mean(self) -> float:
        return self._moments["mean"]

    @property
    def sorting_coefficient(self) -> float:
        return self._moments["std"]

    @property
    def skewness(self) -> float:
        return self._moments["skewness"]

    @property
    def kurtosis(self) -> float:
        return self._moments["kurtosis"]


class ArtificialSample:
    """The class to represent one sample of the artificial dataset."""
    __slots__ = ("_name", "_classes", "_classes_phi", "_distribution", "_components", "_proportions", "_moments")

    def __init__(self, name: str, classes: ndarray, classes_phi: ndarray,
                 distribution: ndarray, components: Sequence[ndarray], proportions: Sequence[float],
                 moments: Tuple[Sequence, Sequence, Sequence, Sequence]):
        """
        Construct an instance of the `ArtificialSample` class.

        **If not necessary, do not manually create the sample, because it will not validate the passed parameters.**

        :param name: The name of this artificial sample. It can be modified.
        :param classes: The grain size classes in microns.
        :param classes_phi: The grain size classes in phi values.
        :param distribution: The frequency distribution of grain size classes.
            Note, the sum of frequencies should be equal to 1.
        :param components: The grain size distributions of all components.
        :param proportions: The proportions of all components.
        :param moments: A tuple that contains the mean, standard deviation, skewness,
            and kurtosis of all components, respectively.
        :return: An instance of the `ArtificialSample` class.
        """
        self._name = name
        self._classes = classes
        self._classes_phi = classes_phi
        self._distribution = distribution
        self._components = components
        self._proportions = proportions
        self._moments = moments

    def __repr__(self):
        return f"AS({self._name}, {len(self._components)})"

    def __len__(self):
        return len(self._components)

    def __iter__(self):
        for i in range(len(self._components)):
            yield self._get_component(i)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get_component(item)
        elif isinstance(item, slice):
            return [self._get_component(index) for index in np.arange(len(self._components))[item]]
        else:
            raise TypeError(f"Component indices must be integers or slices, not {type(item)}.")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        assert isinstance(value, str)
        assert len(value) > 0
        self._name = value

    @property
    def classes(self) -> ndarray:
        return self._classes

    @property
    def classes_phi(self) -> ndarray:
        return self._classes_phi

    @property
    def interval_phi(self) -> float:
        return interval_phi(self._classes_phi)

    @property
    def distribution(self) -> ndarray:
        return self._distribution

    @property
    def sample(self):
        sample = Sample(self._name, self._classes, self._classes_phi, self._distribution)
        return sample

    @property
    def is_valid(self) -> bool:
        valid = True
        for values in [self._proportions, self._components, self._distribution, *self._moments]:
            if np.any(np.logical_or(np.isnan(values), np.isinf(values))):
                valid = False
                break
        return valid

    def _get_component(self, index: int):
        m, std, s, k = self._moments
        component = ArtificialComponent(
            self._classes, self._classes_phi,
            self._components[index], self._proportions[index],
            (m[index], std[index], s[index], k[index]))
        return component


class ArtificialDataset:
    """
    The class to represent one artificial dataset.

    * Get the sample at index i, `sample = dataset[i]`.
    * Iterate all samples, `for sample in dataset`.
    * Iterate partial samples, `for sample in dataset[:10]`.
    * Get the number of samples, `len(dataset)`.
    """
    def __init__(self, parameters: ndarray,
                 distribution_type: DistributionType,
                 min_size=0.02, max_size=2000.0, n_classes=101,
                 precision=4, noise=5):
        """
        Construct an instance of the `ArtificialDataset` class.

        :param parameters: A three-dimensional numpy array that contains the parameters to generate samples.
        :param distribution_type: The type of elementary distribution.
        :param min_size: The minimum grain size class in microns.
        :param max_size: The maximum grain size class in microns.
        :param n_classes: The number of grain size classes. More classes can make the distributions more detailed.
        :param precision: The precision level (decimals) of generated grain size distributions.
        :param noise: The standard deviation of the random noise matrix. Usually, should use `precision+1`.
        :return: An instance of the `ArtificialDataset` class.
        """
        # validations
        assert isinstance(parameters, ndarray)
        assert parameters.ndim == 3
        assert isinstance(min_size, (int, float))
        assert isinstance(max_size, (int, float))
        assert isinstance(n_classes, int)
        assert isinstance(precision, int)
        assert isinstance(noise, int)
        assert min_size > 0
        assert max_size > 0
        assert max_size > min_size
        assert n_classes > 1
        assert precision > 1
        assert noise > 1
        n_samples, n_parameters, n_components = parameters.shape
        distribution_class = get_distribution(distribution_type)
        assert n_parameters == distribution_class.N_PARAMETERS + 1
        assert n_samples > 0
        assert n_parameters == 3 or n_parameters == 4
        assert n_components > 0
        # preparation
        self._name = f"AD({n_samples}, {n_components}, {distribution_type.name})"
        self._parameters = parameters
        self._distribution_type = distribution_type
        self._min_size, self._max_size = min_size, max_size
        self._classes_phi = np.linspace(to_phi(min_size), to_phi(max_size), n_classes)
        self._classes = to_microns(self._classes_phi)
        self._precision = precision
        self._noise = noise
        classes = np.expand_dims(np.expand_dims(self._classes_phi, 0), 0).repeat(n_samples, 0).repeat(n_components, 1)
        proportions, components, (m, std, s, k) = distribution_class.interpret(
            parameters, classes, interval_phi(self._classes_phi))
        noise = np.random.randn(n_samples, n_classes) * (10 ** (-noise))
        distributions = np.round((proportions @ components).squeeze(1) + noise, decimals=precision)
        self._proportions = proportions
        self._components = components
        self._distributions = distributions
        self._mean = m
        self._std = std
        self._skewness = s
        self._kurtosis = k

    def __repr__(self):
        return f"AD({self._parameters.shape[0]}, {self._parameters.shape[2]}, {self._distribution_type.name})"

    def __len__(self):
        return self._parameters.shape[0]

    def __iter__(self):
        for i in range(self._parameters.shape[0]):
            yield self._get_sample(i)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get_sample(item)
        elif isinstance(item, slice):
            return [self._get_sample(index) for index in np.arange(self._parameters.shape[0])[item]]
        else:
            raise TypeError(f"Sample indices must be integers or slices, not {type(item)}.")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        assert isinstance(value, str)
        assert len(value) > 0
        self._name = value

    @property
    def sample_names(self) -> List[str]:
        return [f"AS{i+1}" for i in range(self._parameters.shape[0])]

    @property
    def parameters(self) -> ndarray:
        return self._parameters

    @property
    def distribution_type(self) -> DistributionType:
        return self._distribution_type

    @property
    def n_samples(self) -> int:
        return self._parameters.shape[0]

    @property
    def n_parameters(self) -> int:
        return self._parameters.shape[1]

    @property
    def n_components(self) -> int:
        return self._parameters.shape[2]

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    @property
    def classes(self) -> ndarray:
        return self._classes

    @property
    def classes_phi(self) -> ndarray:
        return self._classes_phi

    @property
    def interval_phi(self) -> float:
        return interval_phi(self._classes_phi)

    @property
    def noise(self) -> int:
        return self._noise

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def distributions(self) -> ndarray:
        return self._distributions

    @property
    def components(self) -> ndarray:
        return self._components

    @property
    def proportions(self) -> ndarray:
        return self._proportions

    @property
    def dataset(self) -> Dataset:
        sample_names = [f"AS{i + 1}" for i in range(self.n_samples)]
        dataset = Dataset("Artificial Dataset", sample_names, self._classes, self._distributions)
        return dataset

    def _get_sample(self, index: int) -> ArtificialSample:
        name = f"AS{index + 1}"
        distribution = self._distributions[index]
        proportions = self._proportions[index, 0, :]
        components = self._components[index]
        moments = self._mean[index], self._std[index], self._skewness[index], self._kurtosis[index]
        sample = ArtificialSample(
            name, self._classes, self._classes_phi,
            distribution, components, proportions, moments)
        return sample
