import numpy as np
import pytest
from scipy.special import softmax

from QGrain.models.dataset import validate_distributions, validate_classes, Dataset, Sample


class TestValidateClasses:
    valid_data: np.ndarray = np.logspace(0, 5, 101) * 0.02

    def test_valid(self):
        valid, array_or_msg = validate_classes(self.valid_data)
        assert valid
        assert isinstance(array_or_msg, np.ndarray)
        assert array_or_msg.ndim == 1

    def test_np_float32(self):
        valid, _ = validate_classes(self.valid_data.astype(np.float32))
        assert valid

    def test_sliced(self):
        valid, _ = validate_classes(self.valid_data[:-1])
        assert valid

    def test_tuple(self):
        valid, _ = validate_classes(tuple(self.valid_data))
        assert valid

    def test_list(self):
        valid, _ = validate_classes(list(self.valid_data))
        assert valid

    def test_with_int(self):
        with_int = self.valid_data.tolist()
        with_int[-1] = 2000
        valid, _ = validate_classes(with_int)
        assert valid

    def test_type_str(self):
        valid, array_or_msg = validate_classes(self.valid_data.astype(str).tolist())
        assert valid
        assert isinstance(array_or_msg, np.ndarray)
        assert array_or_msg.ndim == 1
        assert array_or_msg.shape == self.valid_data.shape
        assert np.all(np.less(np.abs(array_or_msg - self.valid_data), 1e-4))

    def test_type_none(self):
        valid, msg = validate_classes(None)
        assert isinstance(msg, str)
        print("\n", "While passing `None`, the error messages:", msg, end="\n")
        assert not valid

    def test_empty(self):
        valid, msg = validate_classes([])
        assert isinstance(msg, str)
        print("\n", "While passing empty data, the error messages:", msg, end="\n")
        assert not valid

    def test_ndim_2(self):
        data = np.linspace(0, 1, 12).reshape(2, -1)
        valid, msg = validate_classes(data)
        print("\n", "While passing two-dimensional data, the error messages:", msg, end="\n")
        assert not valid

    def test_ndim_3(self):
        data = np.linspace(0, 1, 12).reshape((2, 2, -1))
        valid, msg = validate_classes(data)
        print("\n", "While passing three-dimensional data, the error messages:", msg, end="\n")
        assert not valid

    def test_has_nan(self):
        data = self.valid_data.tolist()
        data[0] = np.nan
        data[10] = np.nan
        data[14] = np.nan
        valid, msg = validate_classes(data)
        print("\n", "While it has 3 NaN values, the error messages:", msg, end="\n")
        assert not valid

    def test_not_incremental_left(self):
        data = self.valid_data.tolist()
        data[0] = 1e4
        valid, msg = validate_classes(data)
        print("\n", "While it's incremental at left end, the error messages:", msg, end="\n")
        assert not valid

    def test_not_incremental_middle(self):
        data = self.valid_data.tolist()
        data[5], data[6] = data[6], data[5]
        valid, msg = validate_classes(data)
        print("\n", "While it's incremental at middle, the error messages:", msg, end="\n")
        assert not valid

    def test_not_incremental_right(self):
        data = self.valid_data.tolist()
        data[-1] = 1e-4
        valid, msg = validate_classes(data)
        print("\n", "While it's incremental at right end, the error messages:", msg, end="\n")
        assert not valid

    def test_quasi_evenly_spaced(self):
        data = np.round(self.valid_data, 3)
        valid, _ = validate_classes(data)
        assert valid

    def test_not_evenly_spaced(self):
        data = np.linspace(1, 100, 101)
        valid, msg = validate_classes(data)
        print("\n", "While it's not evenly spaced, the error messages:", msg, end="\n")
        assert not valid


class TestValidateDistributions:
    valid_data: np.ndarray = softmax(np.random.randn(100, 101), axis=1)

    def test_valid(self):
        valid, array_or_msg = validate_distributions(self.valid_data.tolist())
        assert valid
        assert isinstance(array_or_msg, np.ndarray)
        assert array_or_msg.ndim == 2
        assert np.all(np.less(np.sum(array_or_msg, axis=1) - 1.0, 1e-4))

    def test_np_float32(self):
        valid, _ = validate_distributions(self.valid_data.astype(np.float32))
        assert valid

    def test_sliced(self):
        valid, _ = validate_distributions(self.valid_data[:-1])
        assert valid

    def test_list(self):
        valid, _ = validate_distributions(self.valid_data.tolist())
        assert valid

    def test_with_int(self):
        with_int = self.valid_data.tolist()
        with_int[0][1] += with_int[0][0]
        with_int[0][0] = 0
        valid, _ = validate_distributions(with_int)
        assert valid

    def test_type_str(self):
        valid, array_or_msg = validate_distributions(self.valid_data.astype(str).tolist())
        assert valid
        assert isinstance(array_or_msg, np.ndarray)
        assert array_or_msg.ndim == 2
        assert array_or_msg.shape == self.valid_data.shape
        assert np.all(np.less(np.abs(array_or_msg - self.valid_data), 1e-4))

    def test_type_none(self):
        valid, msg = validate_distributions(None)
        assert isinstance(msg, str)
        print("\n", "While passing `None`, the error messages:", msg, end="\n")
        assert not valid

    def test_empty(self):
        valid, msg = validate_distributions([[]])
        assert isinstance(msg, str)
        print("\n", "While passing empty data, the error messages:", msg, end="\n")
        assert not valid

    def test_ndim_1(self):
        data = np.linspace(0, 1, 12).reshape(-1)
        valid, msg = validate_distributions(data)
        print("\n", "While passing one-dimensional data, the error messages:", msg, end="\n")
        assert not valid

    def test_ndim_3(self):
        data = np.linspace(0, 1, 12).reshape((2, 2, -1))
        valid, msg = validate_distributions(data)
        print("\n", "While passing three-dimensional data, the error messages:", msg, end="\n")
        assert not valid

    def test_has_nan(self):
        data = self.valid_data.copy()
        data[:3, -5:] = np.nan
        valid, msg = validate_distributions(data)
        print("\n", "While it has NaN values, the error messages:", msg, end="\n")
        assert not valid

    def test_not_sum_to_one(self):
        data = self.valid_data.copy()
        data[:3, -5:] = 1e3
        valid, msg = validate_distributions(data)
        print("\n", "While the sums of some rows are not equal to 1, the error messages:", msg, end="\n")
        assert not valid


class TestGrainSizeDataset:
    name = "Test"
    sample_names = [f"Sample_{i + 1}" for i in range(100)]
    classes: np.ndarray = np.logspace(0, 5, 101) * 0.02
    distributions: np.ndarray = softmax(np.random.randn(100, 101), axis=1)
    dataset = Dataset(name, sample_names, classes, distributions)

    def test_get_item(self):
        sample = self.dataset[0]
        assert isinstance(sample, Sample)

    def test_reference(self):
        assert id(self.dataset[0].classes) == id(self.dataset.classes)
        assert id(self.dataset[0].classes_phi) == id(self.dataset.classes_phi)

    def test_iter(self):
        for i, sample in enumerate(self.dataset):
            assert isinstance(sample, Sample)
            assert sample.name == self.dataset.sample_names[i]

    def test_reverse_iter(self):
        for i, sample in enumerate(reversed(self.dataset)):
            assert isinstance(sample, Sample)
            assert sample.name == self.dataset.sample_names[-i - 1]

    def test_next(self):
        iterator = iter(self.dataset)
        i = 0
        try:
            while True:
                sample = next(iterator)
                assert sample.name == self.dataset.sample_names[i]
                i += 1
        except StopIteration:
            assert i == len(self.dataset)

    def test_slice(self):
        for name, sample in zip(self.dataset.sample_names[3:], self.dataset[3:]):
            assert name == sample.name

    def test_consistent(self):
        for i, sample in enumerate(self.dataset):
            assert sample.distribution.tobytes() == self.dataset.distributions[i].tobytes()

    def test_ctor_error(self):
        with pytest.raises(ValueError):
            Dataset("", self.sample_names, self.classes, self.distributions)

        with pytest.raises(ValueError):
            Dataset(self.name, self.sample_names, np.zeros_like(self.classes), self.distributions)

        with pytest.raises(ValueError):
            Dataset(self.name, self.sample_names, self.classes, np.zeros_like(self.distributions))

    def test_index_error(self):
        with pytest.raises(TypeError):
            self.dataset["Sample1"]

        with pytest.raises(TypeError):
            self.dataset[:, 0]


if __name__ == "__main__":
    pytest.main(["-s"])
