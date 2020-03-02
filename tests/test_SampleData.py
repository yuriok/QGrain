import os
import sys
import unittest

import numpy as np

from QGrain.models.SampleData import *


# the properties should be read-only to avoid the modification by mistake
class TestSampleData(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0.1, 10, 100)
        y = x**2
        self.sample = SampleData("test", x, y)

    def tearDown(self):
        self.sample = None

    def test_uuid(self):
        uuid = self.sample.uuid
        with self.assertRaises(AttributeError):
            self.sample.uuid = None

    def test_name(self):
        name = self.sample.name
        with self.assertRaises(AttributeError):
            self.sample.name = None

    def test_classes(self):
        classes = self.sample.classes
        with self.assertRaises(AttributeError):
            self.sample.classes = None

    def test_distribution(self):
        distribution = self.sample.distribution
        with self.assertRaises(AttributeError):
            self.sample.distribution = None

    def test_tag(self):
        tag = self.sample.tag
        with self.assertRaises(AttributeError):
            self.sample.tag = None

    def test_ignore(self):
        self.assertEqual(self.sample.tag, SampleTag.Default)
        self.sample.ignore()
        self.assertEqual(self.sample.tag, SampleTag.Ignored)

    def test_reset_tag(self):
        self.assertEqual(self.sample.tag, SampleTag.Default)
        self.sample.ignore()
        self.assertNotEqual(self.sample.tag, SampleTag.Default)
        self.sample.reset_tag()
        self.assertEqual(self.sample.tag, SampleTag.Default)


if __name__ == "__main__":
    unittest.main()
