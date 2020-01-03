import os
import sys
import unittest

import numpy as np
import xlrd

sys.path.append(os.getcwd())
from models.DataWriter import *


class TestColumnToChar(unittest.TestCase):
    def test_0_to_A(self):
        self.assertEqual(column_to_char(0), "A")

    def test_25_to_Z(self):
        self.assertEqual(column_to_char(25), "Z")

    def test_26_to_AA(self):
        self.assertEqual(column_to_char(26), "AA")


class TestToCellName(unittest.TestCase):
    def test_0_1_to_A1(self):
        self.assertEqual(to_cell_name(0, 0), "A1")

    def test_5_1_to_A6(self):
        self.assertEqual(to_cell_name(5, 0), "A6")

    def test_42_26_to_AA43(self):
        self.assertEqual(to_cell_name(42, 26), "AA43")

class TestDataWriter(unittest.TestCase):
    def setUp(self):
        pass



if __name__ == "__main__":
    unittest.main()
