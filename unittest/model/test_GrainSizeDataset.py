import numpy as np
import pytest
from QGrain.model import *
from scipy.stats import norm


def get_interval_φ(classes_φ: np.ndarray)-> float:
    return abs((classes_φ[0]-classes_φ[-1]) / (len(classes_φ)-1))


class TestGrainSizeDataset:
    def test_ctor(self):
        n_samples = 100
        name_prefix = "Test"
        classes_μm = np.logspace(0, 5, 101) * 0.02
        classes_φ = -np.log2(classes_μm / 1000)
        interval_φ = get_interval_φ(classes_φ)
        names = []
        distributions = []
        for i in range(n_samples):
            names.append(f"{name_prefix}_{i+1}")
            distribution = norm.pdf(classes_φ, loc=5.0+np.random.random(), scale=2.0) * interval_φ
            distributions.append(distribution)
        dataset = GrainSizeDataset()
        dataset.add_batch(classes_μm, names, distributions)
        assert dataset.n_samples == 100

    def test_is_incremental(self):
        classes_μm = np.logspace(0, 5, 101) * 0.02
        assert GrainSizeDataset.is_incremental(classes_μm)
        classes_μm[5] = -1
        assert not GrainSizeDataset.is_incremental(classes_μm)

    def test_validate_classes_μm(self):
        classes_μm = np.logspace(0, 5, 101) * 0.02
        GrainSizeDataset.validate_classes_μm(classes_μm)
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_classes_μm(None)
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_classes_μm(list(classes_μm))
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_classes_μm(classes_μm.astype(np.float32))
        with pytest.raises(ArrayEmptyError):
            GrainSizeDataset.validate_classes_μm(np.array([]))
        with pytest.raises(NaNError):
            classes_μm = np.logspace(0, 5, 101) * 0.02
            classes_μm[0] = np.nan
            GrainSizeDataset.validate_classes_μm(classes_μm)
        with pytest.raises(ClassesNotIncrementalError):
            classes_μm = np.logspace(0, 5, 101) * 0.02
            classes_μm[10] = -1
            GrainSizeDataset.validate_classes_μm(classes_μm)

    def test_validate_sample_name(self):
        GrainSizeDataset.validate_sample_name("Test Sample")
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_sample_name(None)
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_sample_name(1)
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_sample_name(1.0)

    def test_validate_distribution(self):
        classes_μm = np.logspace(0, 5, 101) * 0.02
        classes_φ = -np.log2(classes_μm / 1000)
        interval_φ = get_interval_φ(classes_φ)
        distribution = norm.pdf(classes_φ, loc=5.0+np.random.random(), scale=2.0) * interval_φ
        GrainSizeDataset.validate_distribution(distribution)
        GrainSizeDataset.validate_distribution(distribution*100.0)
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_distribution(None)
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_distribution(list(distribution))
        with pytest.raises(AssertionError):
            GrainSizeDataset.validate_distribution(distribution.astype(np.float32))
        with pytest.raises(ArrayEmptyError):
            GrainSizeDataset.validate_distribution(np.array([]))
        with pytest.raises(NaNError):
            nan_distribution = np.copy(distribution)
            nan_distribution[0] = np.nan
            GrainSizeDataset.validate_distribution(nan_distribution)
        with pytest.raises(DistributionSumError):
            GrainSizeDataset.validate_distribution(distribution*2.0)

    def test_are_classes_match(self):
        classes_μm_1 = np.logspace(0, 5, 101) * 0.02
        classes_μm_2 = np.logspace(0, 5, 80) * 0.02
        classes_μm_3 = np.logspace(0, 4, 80) * 0.02
        assert GrainSizeDataset.are_classes_match(classes_μm_1, classes_μm_1)
        assert not GrainSizeDataset.are_classes_match(classes_μm_1, classes_μm_2)
        assert not GrainSizeDataset.are_classes_match(classes_μm_2, classes_μm_3)

    def test_add_batch_success(self):
        n_samples = 100
        name_prefix = "Test"
        classes_μm = np.logspace(0, 5, 101) * 0.02
        classes_φ = -np.log2(classes_μm / 1000)
        interval_φ = get_interval_φ(classes_φ)
        names = []
        distributions = []
        for i in range(n_samples):
            names.append(f"{name_prefix}_{i+1}")
            distribution = norm.pdf(classes_φ, loc=5.0+np.random.random(), scale=2.0) * interval_φ
            distributions.append(distribution)
        dataset = GrainSizeDataset()
        dataset.add_batch(classes_μm, names, distributions)
        dataset.add_batch(classes_μm, names, distributions)
        assert dataset.n_samples == n_samples*2

    def test_add_batch_failed(self):
        n_samples = 100
        classes_μm_1 = np.logspace(0, 5, 101) * 0.02
        classes_φ_1 = -np.log2(classes_μm_1 / 1000)
        interval_φ_1 = get_interval_φ(classes_φ_1)
        names_1 = []
        distributions_1 = []
        for i in range(n_samples):
            names_1.append(f"Group_1_{i+1}")
            distribution = norm.pdf(classes_φ_1, loc=5.0+np.random.random(), scale=2.0) * interval_φ_1
            distributions_1.append(distribution)

        classes_μm_2 = np.logspace(0, 5, 80) * 0.02
        classes_φ_2 = -np.log2(classes_μm_2 / 1000)
        interval_φ_2 = get_interval_φ(classes_φ_2)
        names_2 = []
        distributions_2 = []
        for i in range(n_samples):
            names_2.append(f"Group_2_{i+1}")
            distribution = norm.pdf(classes_φ_2, loc=5.0+np.random.random(), scale=2.0) * interval_φ_2
            distributions_2.append(distribution)

        dataset = GrainSizeDataset()
        dataset.add_batch(classes_μm_1, names_1, distributions_1)
        with pytest.raises(ClassesNotMatchError):
            dataset.add_batch(classes_μm_2, names_2, distributions_2)

    def test_combine_success(self):
        n_samples = 100
        name_prefix = "Test"
        classes_μm = np.logspace(0, 5, 101) * 0.02
        classes_φ = -np.log2(classes_μm / 1000)
        interval_φ = get_interval_φ(classes_φ)
        names = []
        distributions = []
        for i in range(n_samples):
            names.append(f"{name_prefix}_{i+1}")
            distribution = norm.pdf(classes_φ, loc=5.0+np.random.random(), scale=2.0) * interval_φ
            distributions.append(distribution)
        dataset_1 = GrainSizeDataset()
        dataset_1.add_batch(classes_μm, names, distributions)
        dataset_2 = GrainSizeDataset()
        dataset_2.add_batch(classes_μm, names, distributions)
        assert dataset_1.n_samples == n_samples
        assert dataset_2.n_samples == n_samples
        dataset_1.combine(dataset_2)
        assert dataset_1.n_samples == n_samples*2

    def test_combine_failed(self):
        n_samples = 100
        classes_μm_1 = np.logspace(0, 5, 101) * 0.02
        classes_φ_1 = -np.log2(classes_μm_1 / 1000)
        interval_φ_1 = get_interval_φ(classes_φ_1)
        names_1 = []
        distributions_1 = []
        for i in range(n_samples):
            names_1.append(f"Group_1_{i+1}")
            distribution = norm.pdf(classes_φ_1, loc=5.0+np.random.random(), scale=2.0) * interval_φ_1
            distributions_1.append(distribution)

        classes_μm_2 = np.logspace(0, 5, 80) * 0.02
        classes_φ_2 = -np.log2(classes_μm_2 / 1000)
        interval_φ_2 = get_interval_φ(classes_φ_2)
        names_2 = []
        distributions_2 = []
        for i in range(n_samples):
            names_2.append(f"Group_2_{i+1}")
            distribution = norm.pdf(classes_φ_2, loc=5.0+np.random.random(), scale=2.0) * interval_φ_2
            distributions_2.append(distribution)

        dataset_1 = GrainSizeDataset()
        dataset_1.add_batch(classes_μm_1, names_1, distributions_1)
        dataset_2 = GrainSizeDataset()
        dataset_2.add_batch(classes_μm_2, names_2, distributions_2)
        with pytest.raises(ClassesNotMatchError):
            dataset_1.combine(dataset_2)

    def test_get_sample_by_id(self):
        n_samples = 100
        name_prefix = "Test"
        classes_μm = np.logspace(0, 5, 101) * 0.02
        classes_φ = -np.log2(classes_μm / 1000)
        interval_φ = get_interval_φ(classes_φ)
        names = []
        distributions = []
        for i in range(n_samples):
            names.append(f"{name_prefix}_{i+1}")
            distribution = norm.pdf(classes_φ, loc=5.0+np.random.random(), scale=2.0) * interval_φ
            distributions.append(distribution)
        dataset = GrainSizeDataset()
        dataset.add_batch(classes_μm, names, distributions)

        ids = [sample.uuid for sample in dataset.samples]
        for uuid in ids:
            sample = dataset.get_sample_by_id(uuid)
            assert sample.uuid == uuid
