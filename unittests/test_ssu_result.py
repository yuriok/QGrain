import pytest
from numpy import ndarray

from QGrain.distributions import DistributionType
from QGrain.generate import random_dataset, SIMPLE_PRESET
from QGrain.models.ssu_result import *
from QGrain.ssu import built_in_losses


class TestSSUResult:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)

    def test_ctor(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        assert result.name == as0.name
        assert result.sample is as0
        assert result.n_iterations == self.dataset.n_samples

    def test_get_item(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        sample = result[0]
        assert isinstance(sample, SSUResultComponent)

    def test_iter(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        for i, component in enumerate(result):
            assert isinstance(component, SSUResultComponent)

    def test_reverse_iter(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        for i, component in enumerate(reversed(result)):
            assert isinstance(component, SSUResultComponent)

    def test_slice(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        for component in result[1:]:
            assert isinstance(component, SSUResultComponent)

    def test_none_error(self):
        parameters = self.dataset.parameters.copy()
        with pytest.raises(AssertionError):
            SSUResult(None, self.dataset.distribution_type, parameters[0], parameters, 1.0)
        with pytest.raises(AssertionError):
            SSUResult(self.dataset[0], None, parameters[0], parameters, 1.0)
        with pytest.raises(AssertionError):
            SSUResult(self.dataset[0], self.dataset.distribution_type, None, parameters, 1.0)
        with pytest.raises(AssertionError):
            SSUResult(self.dataset[0], self.dataset.distribution_type, parameters[0], None, 1.0)
        with pytest.raises(AssertionError):
            SSUResult(self.dataset[0], self.dataset.distribution_type, parameters[0], parameters, None)

    def test_ndim_error(self):
        parameters = self.dataset.parameters.copy()
        with pytest.raises(AssertionError):
            SSUResult(self.dataset[0], self.dataset.distribution_type, parameters[0], parameters[0], 1.0)

    def test_n_parameters_error(self):
        parameters = self.dataset.parameters.copy()
        with pytest.raises(AssertionError):
            SSUResult(self.dataset[0], DistributionType.Normal, parameters[0], parameters, 1.0)

    def test_index_error(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        with pytest.raises(TypeError):
            result["C1"]
        with pytest.raises(TypeError):
            result[:, 0]

    def test_apis(self):
        # `ArtificialSample` has similar apis with `SSUResult`, to make them can be used in plotting chart
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        apis = ["name", "classes", "classes_phi", "distribution", "sample", ("sample", "distribution"), "is_valid"]
        component_apis = ["classes", "classes_phi", "distribution", "proportion",
                          "mean", "sorting_coefficient", "skewness", "kurtosis"]
        for api in apis:
            if isinstance(api, tuple):
                api, sub_api = api
                hasattr(getattr(result, api), sub_api)
                hasattr(getattr(as0, api), sub_api)
            assert hasattr(result, api)
            assert hasattr(as0, api)
        for api in component_apis:
            for component in result:
                hasattr(component, api)
            for component in as0:
                hasattr(component, api)

    def test_history(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        result_bytes = result.distribution.tobytes()
        for h in result.history:
            assert isinstance(h, SSUResult)
            assert h is not result
            # will not modify the data of original object
            assert result.distribution.tobytes() == result_bytes

    def test_loss(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        for name in built_in_losses:
            loss = result.loss(name)
            assert isinstance(loss, float)

    def test_loss_series(self):
        as0 = self.dataset[0]
        result = SSUResult(as0, self.dataset.distribution_type,
                           self.dataset.parameters[0], self.dataset.parameters, 1.0)
        for name in built_in_losses:
            loss_series = result.loss_series(name)
            assert isinstance(loss_series, ndarray)
            assert len(loss_series) == result.n_iterations


if __name__ == "__main__":
    pytest.main(["-s"])
