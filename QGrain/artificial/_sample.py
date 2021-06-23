__all__ = ["ComponentParameter",
           "GenerateParameter",
           "ArtificialComponent",
           "ArtificialSample",
           "ArtificialDataset"]

import typing

import numpy as np
from QGrain.models.GrainSizeSample import GrainSizeSample
from QGrain.ssu import SSUViewModel
from QGrain.statistic import convert_μm_to_φ, convert_φ_to_μm
from scipy.stats import skewnorm


class ComponentParameter:
    __slots__ = ("shape", "loc", "scale", "weight")

    def __init__(self, shape: float, loc: float, scale: float, weight: float):
        self.shape = shape
        self.loc = loc
        self.scale = scale
        self.weight = weight


class GenerateParameter:
    __slots__ = ("n_components", "components", "fractions", "func_args")

    def __init__(self, params: np.ndarray):
        self.n_components, left = divmod(len(params), 4)
        assert left == 0
        self.components = [ComponentParameter(*params[i*3:(i+1)*3], params[-self.n_components+i]) for i in range(self.n_components)]
        total_weight = sum([component.weight for component in self.components])
        self.fractions = [component.weight/total_weight for component in self.components]
        self.func_args = []
        for component in self.components:
            self.func_args.append(component.shape)
            self.func_args.append(component.loc)
            self.func_args.append(component.scale)
        self.func_args.extend(self.fractions)
        self.func_args = np.array(self.func_args[:-1])

class ArtificialComponent:
    def __init__(self, classes_μm: np.ndarray, classes_φ: np.ndarray,
                 distribution: np.ndarray,
                 fraction: float):
        assert distribution is not None
        assert isinstance(distribution, np.ndarray)
        assert len(distribution) == len(classes_μm)
        assert 0.0 <= fraction <= 1.0
        self.__distribution = distribution
        self.__fraction = fraction

    @property
    def distribution(self) -> np.ndarray:
        return self.__distribution

    @property
    def fraction(self) -> float:
        return self.__fraction


class ArtificialSample:
    def __init__(self, name: str, classes_μm: np.ndarray, classes_φ: np.ndarray,
                 mixed: np.ndarray,
                 distributions: typing.List[np.ndarray],
                 fractions: typing.List[float], parameter: GenerateParameter):
        assert name is not None
        assert classes_μm is not None
        assert classes_φ is not None
        assert mixed is not None
        assert isinstance(name, str)
        assert isinstance(classes_μm, np.ndarray)
        assert isinstance(classes_φ, np.ndarray)
        assert isinstance(mixed, np.ndarray)
        assert len(classes_μm) != 0
        assert len(classes_μm) == len(classes_φ)
        assert len(classes_μm) == len(mixed)
        assert len(distributions) == len(fractions)
        self.__name = name
        self.__classes_μm = classes_μm
        self.__classes_φ = classes_φ
        self.__distribution = mixed
        self.__components = []
        for distribution, fraction in zip(distributions, fractions):
            component = ArtificialComponent(classes_μm, classes_φ, distribution, fraction)
            self.__components.append(component)
        self.__parameter = parameter


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
    def parameter(self) -> GenerateParameter:
        return self.__parameter

    @property
    def sample_to_fit(self):
        sample = GrainSizeSample(self.name, self.classes_μm, self.classes_φ, self.distribution)
        return sample

    @property
    def view_model(self):
        distributions = [component.distribution for component in self.components]
        fractions = [component.fraction for component in self.components]

        model = SSUViewModel(self.classes_φ, self.distribution,
                                        self.distribution, distributions, fractions,
                                        title=self.name, component_prefix="AC")
        return model


class ArtificialDataset:
    # 1. The `param_array` is a [n_samples x (n_components * 4)] matrix
    # 2. Each row of this matrix is the param list of each sample
    # 3. The order of param list is shape_1, loc_1, scale_1, ..., shape_i, loc_i, scale_i, weight_1, ... weight_i
    #        where i is the index of components
    # 4. Doing softmax algorithm to `param_array[row, -n_components:]` can transfer the weights to actual fractions
    def __init__(self, param_array: np.ndarray,
                 min_μm=0.02, max_μm=2000.0, n_classes=101,
                 precision=4, noise=5):
        # do validations
        assert isinstance(param_array, np.ndarray)
        assert isinstance(min_μm, (int, float))
        assert isinstance(max_μm, (int, float))
        assert isinstance(n_classes, int)
        assert isinstance(precision, int)
        assert isinstance(noise, int)
        assert param_array.ndim == 2
        assert min_μm > 0
        assert max_μm > 0
        assert max_μm > min_μm
        assert n_classes > 1
        assert precision > 1
        assert noise > 1
        # prepare data
        self.__params_array = param_array
        self.__n_samples, n_params = param_array.shape
        self.__n_components, left = divmod(n_params, 4)
        self.__n_classes = n_classes
        assert left == 0
        self.__min_μm, self.__max_μm = min_μm, max_μm
        self.__min_φ, self.__max_φ = convert_μm_to_φ(min_μm), convert_μm_to_φ(max_μm)
        self.__classes_φ = np.linspace(self.__min_φ, self.__max_φ, n_classes)
        self.__interval_φ = abs((self.__max_φ - self.__min_φ) / (n_classes - 1))
        self.__classes_μm = convert_φ_to_μm(self.__classes_φ)
        self.__precision = precision
        self.__noise = noise

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

    @staticmethod
    def get_generate_parameter(params: np.ndarray):
        parameter = GenerateParameter(params)
        return parameter

    def get_sample_by_params(self, name: str, params: np.ndarray):
        fractions = params[-self.n_components:] / np.sum(params[-self.n_components:])
        distributions = np.array([skewnorm.pdf(self.classes_φ, *params[i*3:(i+1)*3]) * self.interval_φ for i in range(self.n_components)])
        mixed = (fractions.reshape((1, -1)) @ distributions).reshape((-1))
        mixed += np.random.random(mixed.shape) * 10 ** (-self.noise)
        mixed = np.round(mixed, self.precision)
        parameter = self.get_generate_parameter(params)
        sample = ArtificialSample(name, self.classes_μm, self.classes_φ,
                                  mixed, distributions, fractions, parameter)
        return sample

    def set_data(self, params_array: np.ndarray):
        self.__params_array = params_array

    def get_sample(self, index: int):
        return self.get_sample_by_params(f"Artificial Sample {index+1}", self.__params_array[index])

    def get_parameter(self, index: int):
        return self.get_generate_parameter(self.__params_array[index])
