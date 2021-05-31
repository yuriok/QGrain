__all__ = ["ReadFileType",
           "get_type_by_name",
           "XLRDError",
           "ValueNotNumberError",
           "CSVEncodingError",
           "DatasetLoader"]

import csv

from typing import Iterable, Tuple

import numpy as np
from xlrd import XLRDError, open_workbook

from QGrain.models.DataLayoutSetting import DataLayoutError, DataLayoutSetting
from QGrain.models.GrainSizeDataset import GrainSizeDataset


class FileTypeNotSupportedError(Exception):
    "Raise while the file type is not supported."
    pass

class ValueNotNumberError(Exception):
    "Raise while the value can not be converted to a number."
    pass

class CSVEncodingError(Exception):
    "Raise while the encoding of csv is not utf-8."
    pass


class DatasetLoader:
    """
    The class to load the grain-size distributions from the loacl files.
    """
    def __init__(self, error_callback, step_callback):
        self.error_callback = error_callback
        self.step_callback = step_callback

    def try_load_data(self, filename: str, file_type: ReadFileType, layout: DataLayoutSetting) -> GrainSizeDataset:
        assert filename is not None
        assert filename != ""

        if file_type == ReadFileType.CSV:
            return self.try_csv(filename, layout)
        elif file_type == ReadFileType.XLS or \
                file_type == ReadFileType.XLSX:
            return self.try_excel(filename, layout)
        else:
            raise NotImplementedError(file_type)

    def try_csv(self, filename: str, layout: DataLayoutSetting) -> GrainSizeDataset:
        try:
            with open(filename, encoding="utf-8") as f:
                r = csv.reader(f)
                raw_data = [row for row in r]
                classes, names, distributions = self.process_raw_data(raw_data, layout)
                dataset = GrainSizeDataset()
                dataset.add_batch(classes, names, distributions)
                return dataset
        except UnicodeDecodeError as e:

            raise CSVEncodingError("The encoding of csv file must be utf-8.") from e

    def try_excel(self, filename: str, layout: DataLayoutSetting) -> GrainSizeDataset:
        sheet = open_workbook(filename).sheet_by_index(0)
        raw_data = []
        for i in range(sheet.nrows):
            raw_data.append(sheet.row_values(i))
        classes, names, distributions = self.process_raw_data(raw_data, layout)
        dataset = GrainSizeDataset()
        dataset.add_batch(classes, names, distributions)
        return dataset
