import os
import sys
import unittest

import numpy as np

from QGrain.models.DataLayoutSetting import *


class TestDataLayoutSetting(unittest.TestCase):
    def test_valid_ctor(self):
        layout = DataLayoutSetting()
        layout = DataLayoutSetting(1, 1, 3, 3)

    def test_invalid_type(self):
        with self.assertRaises(AssertionError):
            DataLayoutSetting("0", 0, 1, 1)
        with self.assertRaises(AssertionError):
            DataLayoutSetting(0, "0", 1, 1)
        with self.assertRaises(AssertionError):
            DataLayoutSetting(0, 0, "1", 1)
        with self.assertRaises(AssertionError):
            DataLayoutSetting(0, 0, 1, "1")
        with self.assertRaises(AssertionError):
            DataLayoutSetting(0.0, 0, 1, 1)
        with self.assertRaises(AssertionError):
            DataLayoutSetting(0, 0.0, 1, 1)
        with self.assertRaises(AssertionError):
            DataLayoutSetting(0, 0, 1.0, 1)
        with self.assertRaises(AssertionError):
            DataLayoutSetting(0, 0, 1, 1.0)

    def test_negative(self):
        with self.assertRaises(DataLayoutError):
            DataLayoutSetting(-1, 0, 1, 1)
        with self.assertRaises(DataLayoutError):
            DataLayoutSetting(0, -1, 1, 1)
        with self.assertRaises(DataLayoutError):
            DataLayoutSetting(0, 0, -1, 1)
        with self.assertRaises(DataLayoutError):
            DataLayoutSetting(0, 0, 1, -1)

    def test_unexcepted_smaller(self):
        with self.assertRaises(DataLayoutError):
            DataLayoutSetting(3, 3, 1, 4)
        with self.assertRaises(DataLayoutError):
            DataLayoutSetting(3, 3, 4, 1)

    def test_read_only(self):
        layout = DataLayoutSetting()
        layout.classes_row
        layout.sample_name_column
        layout.distribution_start_row
        layout.distribution_start_column
        with self.assertRaises(AttributeError):
            layout.classes_row = 0
        with self.assertRaises(AttributeError):
            layout.sample_name_column = 0
        with self.assertRaises(AttributeError):
            layout.distribution_start_row = 0
        with self.assertRaises(AttributeError):
            layout.distribution_start_column = 0


if __name__ == "__main__":
    unittest.main()
