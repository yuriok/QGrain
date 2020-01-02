import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())
from models.SampleDataset import *


class TestIsCremental(unittest.TestCase):
    def setUp(self):
        self.dataset = SampleDataset()
    
    def tearDown(self):
        self.dataset = None

    def test_empty(self):
        res = self.dataset.is_incremental([])
        self.assertTrue(res)

    def test_one_element(self):
        for i in range(100):
            res = self.dataset.is_incremental([i])
            self.assertTrue(res)

    def test_all_same(self):
        res = self.dataset.is_incremental(np.ones((100,)))
        self.assertFalse(res)

    def test_one_same(self):
        instance = [0, 1, 2, 5, 5, 6]
        res = self.dataset.is_incremental(instance)
        self.assertFalse(res)

    def test_all_reverse(self):
        res = self.dataset.is_incremental(range(100, 0, -1))
        self.assertFalse(res)

    def test_one_reverse(self):
        instance = [0, 1, 2, 5, 4, 6]
        res = self.dataset.is_incremental(instance)
        self.assertFalse(res)


class TestValidateClasses(unittest.TestCase):
    def setUp(self):
        self.dataset = SampleDataset()
    
    def tearDown(self):
        self.dataset = None

    def test_valid(self):
        self.dataset.validate_classes(np.linspace(0.1, 10, 100, dtype=np.float64))
    
    def test_none(self):
        with self.assertRaises(AssertionError):
            self.dataset.validate_classes(None)

    def test_non_ndarray(self):
        with self.assertRaises(AssertionError):
            self.dataset.validate_classes([0, 1, 2, 3, 4, 5])

    def test_non_float64(self):
        with self.assertRaises(AssertionError):
            self.dataset.validate_classes(np.linspace(0, 100, 100, dtype=np.int32))

    def test_length_zero(self):
        with self.assertRaises(ArrayEmptyError):
            self.dataset.validate_classes(np.ones((0,), dtype=np.float64))
    
    def test_has_nan(self):
        with self.assertRaises(NaNError):
            classes = np.linspace(0.1, 10, 100, dtype=np.float64)
            classes[-1] = np.nan
            self.dataset.validate_classes(classes)

    def test_non_incremental(self):
        with self.assertRaises(ClassesNotIncrementalError):
            self.dataset.validate_classes(np.ones((100,), dtype=np.float64))


class TestValidateSampleName(unittest.TestCase):
    def setUp(self):
        self.dataset = SampleDataset()
    
    def tearDown(self):
        self.dataset = None

    def test_valid(self):
        self.dataset.validate_sample_name("1st Sample")

    def test_none(self):
        with self.assertRaises(AssertionError):
            self.dataset.validate_sample_name(None)

    def test_non_str(self):
        with self.assertRaises(AssertionError):
            self.dataset.validate_sample_name(2)

    def test_empty(self):
        with self.assertRaises(SampleNameEmptyError):
            self.dataset.validate_sample_name("")


class TestValidateDistribution(unittest.TestCase):
    def setUp(self):
        self.dataset = SampleDataset()
    
    def tearDown(self):
        self.dataset = None

    def test_valid(self):
        self.dataset.validate_distribution(np.ones((100,), dtype=np.float64))
    
    def test_none(self):
        with self.assertRaises(AssertionError):
            self.dataset.validate_distribution(None)

    def test_non_ndarray(self):
        with self.assertRaises(AssertionError):
            self.dataset.validate_distribution(list(np.ones((100,), dtype=np.float64)))

    def test_non_float64(self):
        with self.assertRaises(AssertionError):
            self.dataset.validate_distribution(np.ones((100,), dtype=np.int32))

    def test_length_zero(self):
        with self.assertRaises(ArrayEmptyError):
            self.dataset.validate_distribution(np.ones((0,)))
    
    def test_has_nan(self):
        with self.assertRaises(NaNError):
            distribution = np.ones((100,), dtype=np.float64)
            distribution[-1] = np.nan
            self.dataset.validate_distribution(distribution)

    def test_sum_error(self):
        with self.assertRaises(DistributionSumError):
            self.dataset.validate_distribution(np.zeros((100,)))


class TestAddBatch(unittest.TestCase):
    def setUp(self):
        self.dataset = SampleDataset()
    
    def tearDown(self):
        self.dataset = None

    def test_ctor(self):
        self.assertEqual(self.dataset.data_count, 0)
        self.assertFalse(self.dataset.has_data)

    def test_valid_add(self):
        classes = np.linspace(0.1, 10, 100, dtype=np.float64)
        sample_count = 200
        names = ["Sample {0}".format(i+1) for i in range(sample_count)]
        distributions = []
        for i in range(sample_count):
            distribution = np.linspace(0.1, 10, 100, dtype=np.float64)**((i+1)/100)
            distribution = distribution / np.sum(distribution)
            distributions.append(distribution)
        self.dataset.add_batch(classes, names, distributions)
        self.assertEqual(self.dataset.data_count, sample_count)
        self.assertTrue(self.dataset.has_data)

        for sample, name, distribution in zip(self.dataset.samples, names, distributions):
            self.assertIs(sample.classes, classes)
            self.assertIs(sample.distribution, distribution)
            self.assertEqual(sample.name, name)

    def test_is_clean_while_failed(self):
        classes = np.linspace(0.1, 10, 100, dtype=np.float64)
        sample_count = 200
        names = ["Sample {0}".format(i+1) for i in range(sample_count)]
        distributions = []
        for i in range(sample_count):
            distribution = (np.linspace(0.1, 10, 100, dtype=np.float64)**((i+1)/100))
            if i < 50:
                distribution = distribution / np.sum(distribution)
            distributions.append(distribution)
        with self.assertRaises(DistributionSumError):
            self.dataset.add_batch(classes, names, distributions)
        self.assertEqual(self.dataset.data_count, 0)
        self.assertFalse(self.dataset.has_data)

    def test_twice_add(self):
        classes1 =np.linspace(0.1, 10, 100, dtype=np.float64)
        classes2 =np.linspace(1, 10, 100, dtype=np.float64)
        disrtibution = classes1**2
        disrtibution = disrtibution / np.sum(disrtibution)
        self.dataset.add_batch(classes1, ["sample1"], [disrtibution])
        self.assertEqual(self.dataset.data_count, 1)
        self.assertTrue(self.dataset.has_data)

        self.dataset.add_batch(classes1, ["sample2"], [disrtibution])
        self.assertEqual(self.dataset.data_count, 2)
        self.assertTrue(self.dataset.has_data)

        with self.assertRaises(ClassesNotMatchError):
            self.dataset.add_batch(classes2, ["sample3"], [disrtibution])
        self.assertEqual(self.dataset.data_count, 2)
        self.assertTrue(self.dataset.has_data)


if __name__ == "__main__":
    unittest.main()
