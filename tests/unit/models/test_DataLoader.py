import csv
import os
import sys
import unittest
from typing import Iterable
from uuid import uuid4

import numpy as np
import xlsxwriter
import xlwt

from QGrain.models.DataLayoutSettings import *
from QGrain.models.DataLoader import *


class TestProcessRawData(unittest.TestCase):
    def setUp(self):
        self.valid_raw_data = [
            ["Classes", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ["Sample 1", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            ["Sample 2", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            ["Sample 3", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            ["Sample 4", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            ["Sample 5", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
        self.loader = DataLoader()

    def tearDown(self):
        self.valid_raw_data = None
        self.loader = None

    def test_valid(self):
        classes, names, distributions = self.loader.process_raw_data(self.valid_raw_data)
        self.assertEqual(classes.dtype, np.float64)
        self.assertEqual(len(classes), 10)
        self.assertEqual(len(names), 5)
        self.assertEqual(len(distributions), 5)
        for name, distribution in zip(names, distributions):
            self.assertIsNotNone(name)
            self.assertIsNotNone(distribution)
            self.assertEqual(len(distribution), 10)
            self.assertEqual(distribution.dtype, np.float64)

    def test_name_none(self):
        name_none_raw_data = [
            ["Classes", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [None, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
        classes, names, distributions = self.loader.process_raw_data(name_none_raw_data)
        for name in names:
            self.assertIsNotNone(name)

    def test_name_empty(self):
        name_empty_raw_data = [
            ["Classes", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ["", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
        classes, names, distributions = self.loader.process_raw_data(name_empty_raw_data)
        for name in names:
            self.assertNotEqual(name, "")

    # Note: Due to the feature of slice,
    # it will not raise exceptio,
    # if the distribution start row/column is out of range.
    def test_classes_row_out_of_range(self):
        layout = DataLayoutSettings(20, 0, 21, 1)
        with self.assertRaises(DataLayoutError):
            classes, names, distributions = self.loader.process_raw_data(self.valid_raw_data, layout)

    def test_sample_name_column_out_of_range(self):
        layout = DataLayoutSettings(0, 20, 1, 21)
        with self.assertRaises(DataLayoutError):
            classes, names, distributions = self.loader.process_raw_data(self.valid_raw_data, layout)

    def test_classes_convertible_str(self):
        classes_convertible_raw_data = [
            ["Classes", 1, "2", "3", 4, 5, 6, 7, 8, 9, 10],
            ["Sample", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
        classes, names, distributions = self.loader.process_raw_data(classes_convertible_raw_data)
        self.assertFalse(np.any(np.isnan(classes)))

    def test_classes_unconvertible_str(self):
        classes_unconvertible_raw_data = [
            ["Classes", 1, "h", "e", "l", "l", "o", 7, 8, 9, 10],
            ["Sample", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
        with self.assertRaises(ValueNotNumberError):
            classes, names, distributions = self.loader.process_raw_data(classes_unconvertible_raw_data)

    def test_classes_none(self):
        classes_none_raw_data = [
            ["Classes", 1, None, 3, 4, 5, 6, 7, 8, 9, 10],
            ["Sample", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
        classes, names, distributions = self.loader.process_raw_data(classes_none_raw_data)
        self.assertTrue(np.any(np.isnan(classes)))

    def test_distribution_convertible_str(self):
        distribution_convertible_raw_data = [
            ["Classes", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ["Sample", 10, "10", 10, 10, 10, 10, 10, 10, 10, 10]]
        classes, names, distributions = self.loader.process_raw_data(distribution_convertible_raw_data)
        self.assertFalse(np.any(np.isnan(distributions[0])))

    def test_distribution_unconvertible_str(self):
        distribution_unconvertible_raw_data = [
            ["Classes", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ["Sample", 10, "h", "e", "l", "l", "o", 10, 10, 10, 10]]
        with self.assertRaises(ValueNotNumberError):
            classes, names, distributions = self.loader.process_raw_data(distribution_unconvertible_raw_data)

    def test_distribution_none(self):
        distribution_none_raw_data = [
            ["Classes", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ["Sample", 10, None, 10, 10, 10, 10, 10, 10, 10, 10]]
        classes, names, distributions = self.loader.process_raw_data(distribution_none_raw_data)
        self.assertTrue(np.any(np.isnan(distributions[0])))


class TestTryCsvAndExcel(unittest.TestCase):
    def create_csv_file(self, filename: str, data: Iterable[Iterable], encoding="utf-8", dialect="excel"):
        with open(filename, "w", encoding=encoding, newline="") as f:
            writer = csv.writer(f, dialect=dialect)
            writer.writerows(data)

    def create_xls_file(self, filename: str, data: Iterable[Iterable]):
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("Test")
        for row_index, row in enumerate(data):
            for col_index, value in enumerate(row):
                sheet.write(row_index, col_index, value)
        workbook.save(filename)

    def create_xlsx_file(self, filename: str, data: Iterable[Iterable]):
        workbook = xlsxwriter.Workbook(filename)
        sheet = workbook.add_worksheet("Test")
        for row_index, row in enumerate(data):
            for col_index, value in enumerate(row):
                sheet.write(row_index, col_index, value)
        workbook.close()

    def setUp(self):
        self.temp_folder = uuid4().__str__()
        os.mkdir(self.temp_folder)

        self.valid_raw_data = [
            ["Classes", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ["Sample 1", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            ["Sample 2", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            ["样品 3", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            ["Sample 4", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            ["Sample 5", 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]

        self.valid_csv_path = os.path.join(self.temp_folder, uuid4().__str__()+".csv")

        self.gb2312_csv_path = os.path.join(self.temp_folder, uuid4().__str__()+".csv")
        self.excel_tab_dialect_csv_path = os.path.join(self.temp_folder, uuid4().__str__()+".csv")
        self.valid_xls_file = os.path.join(self.temp_folder, uuid4().__str__()+".xls")
        self.valid_xlsx_file = os.path.join(self.temp_folder, uuid4().__str__()+".xlsx")
        self.create_csv_file(self.valid_csv_path, self.valid_raw_data)
        self.create_csv_file(self.gb2312_csv_path, self.valid_raw_data, encoding="gb2312")
        self.create_csv_file(self.excel_tab_dialect_csv_path, self.valid_raw_data, dialect=csv.excel_tab)
        self.create_xls_file(self.valid_xls_file, self.valid_raw_data)
        self.create_xlsx_file(self.valid_xlsx_file, self.valid_raw_data)
        self.loader = DataLoader()

    def tearDown(self):
        os.remove(self.valid_csv_path)
        os.remove(self.gb2312_csv_path)
        os.remove(self.excel_tab_dialect_csv_path)
        os.remove(self.valid_xls_file)
        os.remove(self.valid_xlsx_file)
        os.removedirs(self.temp_folder)
        self.valid_raw_data = None
        self.loader = None

    def test_csv_valid(self):
        dataset = self.loader.try_csv(self.valid_csv_path, DataLayoutSettings())
        self.assertTrue(dataset.has_data)
        self.assertEqual(dataset.data_count, 5)

    def test_csv_non_utf8(self):
        with self.assertRaises(CSVEncodingError):
            dataset = self.loader.try_csv(self.gb2312_csv_path, DataLayoutSettings())

    def test_csv_non_excel_dialect(self):
        with self.assertRaises(Exception):
            dataset = self.loader.try_csv(self.excel_tab_dialect_csv_path, DataLayoutSettings())

    def test_csv_file_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            dataset = self.loader.try_csv(uuid4().__str__(), DataLayoutSettings())

    def test_csv_wrong_file_type(self):
        with self.assertRaises(Exception):
            dataset = self.loader.try_csv(self.valid_xls_file, DataLayoutSettings())

    def test_xls_valid(self):
        dataset = self.loader.try_excel(self.valid_xls_file, DataLayoutSettings())
        self.assertTrue(dataset.has_data)
        self.assertEqual(dataset.data_count, 5)

    def test_xlsx_valid(self):
        dataset = self.loader.try_excel(self.valid_xlsx_file, DataLayoutSettings())
        self.assertTrue(dataset.has_data)
        self.assertEqual(dataset.data_count, 5)

    def test_try_load(self):
        dataset = self.loader.try_load_data(self.valid_csv_path, FileType.CSV, DataLayoutSettings())
        self.assertTrue(dataset.has_data)
        self.assertEqual(dataset.data_count, 5)
        dataset = self.loader.try_load_data(self.valid_xls_file, FileType.XLS, DataLayoutSettings())
        self.assertTrue(dataset.has_data)
        self.assertEqual(dataset.data_count, 5)
        dataset = self.loader.try_load_data(self.valid_xlsx_file, FileType.XLSX, DataLayoutSettings())
        self.assertTrue(dataset.has_data)
        self.assertEqual(dataset.data_count, 5)


if __name__ == "__main__":
    unittest.main()
