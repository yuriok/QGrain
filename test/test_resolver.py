import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())
from resolvers import *


class TestValidateData(unittest.TestCase):

    def test_valid(self):
        x = np.linspace(0, 10, 101)
        y = np.sin(x)
        Resolver.validate_data(x, y)

    def test_x_none(self):
        with self.assertRaises(DataInvalidError):
            Resolver.validate_data(None, np.array([2.1]))

    def test_y_none(self):
        with self.assertRaises(DataInvalidError):
            Resolver.validate_data(np.array([2.1]), None)

    def test_x_type_invalid(self):
        with self.assertRaises(DataInvalidError):
            Resolver.validate_data(1, np.array([2.1]))

    def test_y_type_invalid(self):
        with self.assertRaises(DataInvalidError):
            Resolver.validate_data(np.array([2.1]), [1.1])

    def test_length_invalid(self):
        x = np.linspace(0, 10, 101)
        y = np.square(x)[1:]
        with self.assertRaises(DataInvalidError):
            Resolver.validate_data(x, y)

    def test_x_nan_invalid(self):
        with self.assertRaises(DataInvalidError):
            Resolver.validate_data(np.array([2.1, np.nan]), np.array([1.1, 2.2]))

    def test_y_nan_invalid(self):
        with self.assertRaises(DataInvalidError):
            Resolver.validate_data(np.array([2.1, 3.2]), np.array([np.nan, 2.2]))

if __name__ == "__main__":
    unittest.main()
