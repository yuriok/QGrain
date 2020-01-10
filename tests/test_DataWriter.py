import os
import sys
import unittest

import numpy as np
from scipy.stats import norm
import xlrd

sys.path.append(os.getcwd())
from models.DataWriter import *
from models.SampleData import SampleData
from resolvers.Resolver import Resolver


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

global fitting_result
fitting_result = None
import types
from uuid import uuid4

def success_hook(self, algorithm_result):
    global fitting_result
    fitting_result = self.get_fitting_result(algorithm_result.x)

class TestDataWriter(unittest.TestCase):
    def setUp(self):
        global fitting_result
        resolver = Resolver()
        resolver.component_number = 1
        resolver.distribution_type = DistributionType.Normal
        x = np.linspace(0.1, 10, 100)
        y = norm.pdf(x, 4.21, 2.51)
        resolver.on_fitting_succeeded = types.MethodType(success_hook, resolver)
        prepared = []
        for i in range(5):
            resolver.feed_data(SampleData("Sample" + str(i+1), x, y))
            resolver.try_fit()
            assert fitting_result is not None
            prepared.append(fitting_result)
            fitting_result = None
        self.prepared_results = prepared

        self.temp_folder = uuid4().__str__()
        os.mkdir(self.temp_folder)
        self.filename_list = []

        self.writer = DataWriter()

    def tearDown(self):
        for filename in self.filename_list:
            os.remove(filename)
        os.removedirs(self.temp_folder)

    def test_try_save_as_csv(self):
        filename = os.path.join(self.temp_folder, uuid4().__str__()+".csv")
        self.writer.try_save_as_csv(filename, self.prepared_results)
        self.filename_list.append(filename)

    def test_try_save_as_xls(self):
        filename = os.path.join(self.temp_folder, uuid4().__str__()+".xls")
        self.writer.try_save_as_excel(filename, self.prepared_results, False, False)
        self.filename_list.append(filename)

    def test_try_save_as_xlsx(self):
        filename = os.path.join(self.temp_folder, uuid4().__str__()+".xlsx")
        self.writer.try_save_as_excel(filename, self.prepared_results, True, True)
        self.filename_list.append(filename)

    def test_try_save_file(self):
        csv_filename = os.path.join(self.temp_folder, uuid4().__str__()+".csv")
        self.writer.try_save_data(csv_filename, FileType.CSV, self.prepared_results, False)
        self.filename_list.append(csv_filename)
        xls_filename = os.path.join(self.temp_folder, uuid4().__str__()+".xls")
        self.writer.try_save_data(xls_filename, FileType.XLS, self.prepared_results, False)
        self.filename_list.append(xls_filename)
        xlsx_filename = os.path.join(self.temp_folder, uuid4().__str__()+".xlsx")
        self.writer.try_save_data(xlsx_filename, FileType.XLSX, self.prepared_results, True)
        self.filename_list.append(xlsx_filename)


if __name__ == "__main__":
    unittest.main()
