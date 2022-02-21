import typing

import numpy as np

from ..model import GrainSizeDataset, GrainSizeSample
from ..ssu import DISTRIBUTION_CLASS_MAP, DistributionType, SSUViewModel
from ..statistic import convert_μm_to_φ, convert_φ_to_μm


class ArtificialComponent:
    def __init__(self,
                 distribution: np.ndarray,
                 proportion: float,
                 moments: typing.Tuple[float, float, float, float]):
        self.__distribution = distribution
        self.__proportion = proportion
        m, v, s, k = moments
        self.__moments = dict(mean=m, std=np.sqrt(v), skewness=s, kurtosis=k)

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution

    @property
    def proportion(self) -> float:
        return self.__proportion

    @property
    def moments(self) -> dict:
        return self.__moments


class ArtificialSample:
    def __init__(self,
                 name: str,
                 classes_μm: np.ndarray,
                 classes_φ: np.ndarray,
                 distribution: np.ndarray,
                 components: typing.Iterable[np.ndarray],
                 proportions: typing.Iterable[float],
                 moments: typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        self.__name = name
        self.__classes_μm = classes_μm
        self.__classes_φ = classes_φ
        self.__distribution = distribution
        self.__components = []
        m, v, s, k = moments
        for component, proportion, moments in zip(components, proportions, zip(m, v, s, k)):
            artificial_component = ArtificialComponent(component, proportion, moments)
            self.__components.append(artificial_component)
        self.__components.sort(key=lambda component: component.moments["mean"], reverse=True)

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str):
        assert isinstance(value, str)
        self.__name = value

    @property
    def classes_μm(self) -> np.ndarray:
        return self.__classes_μm

    @property
    def classes_φ(self) -> np.ndarray:
        return self.__classes_φ

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution

    @property
    def n_components(self) -> int:
        return len(self.__components)

    @property
    def components(self) -> typing.List[ArtificialComponent]:
        return self.__components

    @property
    def sample_to_fit(self):
        sample = GrainSizeSample(self.name, self.classes_μm, self.classes_φ, self.distribution)
        return sample

    @property
    def view_model(self):
        distributions = [component.distribution for component in self.components]
        proportions = [component.proportion for component in self.components]

        model = SSUViewModel(self.classes_φ, self.distribution,
                             self.distribution, distributions, proportions,
                             title=self.name, component_prefix="AC")
        return model


class ArtificialDataset:
    def __init__(self,
                 target: typing.Iterable[typing.Iterable[typing.Tuple[float, float]]],
                 distribution_type: DistributionType,
                 n_samples: int,
                 params: np.ndarray = None,
                 min_μm=0.02, max_μm=2000.0, n_classes=101,
                 precision=4, noise=5):
        # do validations
        if params is None:
            params = get_params(target, n_samples)
        else:
            assert isinstance(params, np.ndarray)
            assert params.ndim == 3
            assert params.shape[0] == n_samples
        assert isinstance(min_μm, (int, float))
        assert isinstance(max_μm, (int, float))
        assert isinstance(n_classes, int)
        assert isinstance(precision, int)
        assert isinstance(noise, int)
        assert min_μm > 0
        assert max_μm > 0
        assert max_μm > min_μm
        assert n_classes > 1
        assert precision > 1
        assert noise > 1
        # prepare data
        self.__target = target
        self.__distribution_type = distribution_type
        self.__params = params
        self.__n_samples, self.__n_params, self.__n_components = params.shape
        self.__n_classes = n_classes
        self.__min_μm, self.__max_μm = min_μm, max_μm
        self.__min_φ, self.__max_φ = convert_μm_to_φ(min_μm), convert_μm_to_φ(max_μm)
        self.__classes_φ = np.linspace(self.__min_φ, self.__max_φ, n_classes)
        self.__interval_φ = abs((self.__max_φ - self.__min_φ) / (n_classes - 1))
        self.__classes_μm = convert_φ_to_μm(self.__classes_φ)
        self.__precision = precision
        self.__noise = noise

        classes = np.expand_dims(np.expand_dims(self.classes_φ, 0), 0).repeat(self.n_samples, 0).repeat(self.n_components, 1)
        distribution_class = DISTRIBUTION_CLASS_MAP[distribution_type]
        proportions, components, (m, v, s, k) = distribution_class.interpret(params, classes, self.interval_φ)
        noise = np.random.randn(self.n_samples, self.n_classes) * (10**(-self.noise))
        distributions = np.round((proportions @ components).squeeze(1) + noise, decimals=self.precision)
        self.__distributions = distributions
        self.__samples = []
        for i in range(self.n_samples):
            sample = ArtificialSample(
                f"AS{i+1}",
                self.classes_μm,
                self.classes_φ,
                distributions[i],
                components[i],
                proportions[i, 0],
                (m[i], v[i], s[i], k[i]))
            self.__samples.append(sample)

    @property
    def target(self) -> typing.Iterable[typing.Iterable[typing.Tuple[float, float]]]:
        return self.__target

    @property
    def distribution_type(self) -> DistributionType:
        return self.__distribution_type

    @property
    def n_samples(self) -> int:
        return self.__n_samples

    @property
    def n_components(self) -> int:
        return self.__n_components

    @property
    def n_classes(self) -> int:
        return self.__n_classes

    @property
    def min_μm(self) -> float:
        return self.__min_μm

    @property
    def max_μm(self) -> float:
        return self.__max_μm

    @property
    def min_φ(self) -> float:
        return self.__min_φ

    @property
    def max_φ(self) -> float:
        return self.__max_φ

    @property
    def classes_φ(self) -> np.ndarray:
        return self.__classes_φ

    @property
    def classes_μm(self) -> np.ndarray:
        return self.__classes_μm

    @property
    def interval_φ(self) -> float:
        return self.__interval_φ

    @property
    def noise(self) -> int:
        return self.__noise

    @property
    def precision(self) -> int:
        return self.__precision

    @property
    def params(self) -> np.ndarray:
        return self.__params

    @property
    def samples(self) -> typing.List[ArtificialSample]:
        return self.__samples

    @property
    def distributions(self) -> np.ndarray:
        return self.__distributions

    @property
    def dataset_to_fit(self) -> GrainSizeDataset:
        dataset = GrainSizeDataset()
        sample_names = [f"AS{i+1}" for i in range(self.n_samples)]
        dataset.add_batch(self.classes_μm, sample_names, self.distributions)
        return dataset


def get_params(
    target: typing.Iterable[typing.Iterable[typing.Tuple[float, float]]],
    n_samples: int):
    n_components = len(target)
    n_params = len(target[0])
    params = np.random.randn(n_samples, n_params, n_components)
    for component_i, sub_target in enumerate(target):
        for param_i, (mean, std) in enumerate(sub_target):
            params[:, param_i, component_i] = params[:, param_i, component_i] * std + mean
    return params


def get_dataset(
        target: typing.Iterable[typing.Iterable[typing.Tuple[float, float]]],
        distribution_type: DistributionType,
        n_samples: int,
        min_μm=0.02,
        max_μm=2000.0,
        n_classes=101,
        precision=4,
        noise=5):
    dataset = ArtificialDataset(
        target, distribution_type,
        n_samples, params=None,
        min_μm=min_μm, max_μm=max_μm, n_classes=n_classes,
        precision=precision, noise=noise)
    return dataset


def get_sample(
        target: typing.Iterable[typing.Iterable[typing.Tuple[float, float]]],
        distribution_type: DistributionType,
        min_μm=0.02,
        max_μm=2000.0,
        n_classes=101,
        precision=4,
        noise=5):
    dataset = ArtificialDataset(
        target, distribution_type,
        1, params=None,
        min_μm=min_μm, max_μm=max_μm, n_classes=n_classes,
        precision=precision, noise=noise)
    sample=dataset.samples[0]
    return sample


def get_mean_sample(
        target: typing.Iterable[typing.Iterable[typing.Tuple[float, float]]],
        distribution_type: DistributionType,
        min_μm=0.02,
        max_μm=2000.0,
        n_classes=101,
        precision=4,
        noise=5):
    params = np.expand_dims(np.array([[mean for (mean, std) in comp] for comp in target]).T, 0)
    dataset = ArtificialDataset(
        target, distribution_type,
        1, params=params,
        min_μm=min_μm, max_μm=max_μm, n_classes=n_classes,
        precision=precision, noise=noise)
    sample=dataset.samples[0]
    return sample
