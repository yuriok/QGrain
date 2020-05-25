__all__ = ["FileType", "XLRDError", "ValueNotNumberError", "CSVEncodingError", "DataLoader"]

import csv
from enum import Enum, unique
from typing import Iterable, Tuple

import numpy as np
from xlrd import XLRDError, open_workbook

from QGrain.models.DataLayoutSettings import DataLayoutError, DataLayoutSettings
from QGrain.models.SampleDataset import SampleDataset


class FileType:
    XLS = 0
    XLSX = 1
    CSV = 2

class ValueNotNumberError(Exception):
    "Raise while the value can not be converted to a number."
    pass

class CSVEncodingError(Exception):
    "Raise while the encoding of csv is not utf-8."
    pass


class DataLoader:
    """
    The class to load the grain size distributions from the loacl files.
    """
    def __init__(self):
        pass

    def try_load_data(self, filename: str, file_type: FileType, layout: DataLayoutSettings) -> SampleDataset:
        assert filename is not None
        assert filename != ""

        if file_type == FileType.CSV:
            return self.try_csv(filename, layout)
        elif file_type == FileType.XLS or \
                file_type == FileType.XLSX:
            return self.try_excel(filename, layout)
        else:
            raise NotImplementedError(file_type)

    def process_raw_data(self, raw_data: Iterable[Iterable],
                         assigned_layout: DataLayoutSettings = None,
                         replace_name_none="NONE", replace_name_empty="EMPTY") \
            -> Tuple[np.ndarray, Iterable[str], Iterable[np.ndarray]]:
        """
        Method to convert the raw data table to grain size classes, sample names and distributions.

        Args:
            raw_data: A 2-d `Iterable` object, e.g. List[List].
            assigned_layout: A `DataLayoutSettings` object, by default it will use the parameterless `ctor` of `DataLayoutSettings` to create one.
            replace_name_none: The `str` to replace the `None` value of sample names.
            replace_name_empty: Similar to above.

        Returns:
            A tuple that contains the classes array, sample names and distributions.

        Raises:
            DataLayoutError: If the `classes_row` or `sample_name_column` of the layout setting is out of range.
            ValueNotNumberError: If the value of classes or distributions can not be converted to real number.
        """
        if assigned_layout is None:
            layout = DataLayoutSettings()
        else:
            layout = assigned_layout
        try:
            classes = np.array(raw_data[layout.classes_row][layout.distribution_start_column:], dtype=np.float64)
            names = []
            distributions = []
            for row_values in raw_data[layout.distribution_start_row:]:
                sample_name = row_values[layout.sample_name_column]
                if sample_name is None:
                    sample_name = replace_name_none
                # users may use pure number as the sample name
                elif type(sample_name) != str:
                    sample_name = str(sample_name)
                elif sample_name == "":
                    sample_name = replace_name_empty
                # check if it's a empty row, i.e. the values all are empty string
                is_empty_row = True
                for distribution_value in row_values[layout.distribution_start_column:]:
                    if distribution_value != "":
                        is_empty_row = False
                        break
                # if it's a empty row, jump this row to process the next one
                if is_empty_row:
                    continue
                distribution = np.array(row_values[layout.distribution_start_column:], dtype=np.float64)
                names.append(sample_name)
                distributions.append(distribution)

            return classes, names, distributions
        except IndexError as e:
            raise DataLayoutError("The data layout setting does not match the data file.") from e
        except ValueError as e:
            raise ValueNotNumberError("Some value can not be converted to real number, check the data file and layout setting.") from e

    def try_csv(self, filename: str, layout: DataLayoutSettings) -> SampleDataset:
        try:
            with open(filename, encoding="utf-8") as f:
                r = csv.reader(f)
                raw_data = [row for row in r]
                classes, names, distributions = self.process_raw_data(raw_data, layout)
                dataset = SampleDataset()
                dataset.add_batch(classes, names, distributions)
                return dataset
        except UnicodeDecodeError as e:
            raise CSVEncodingError("The encoding of csv file must be utf-8.") from e

    def try_excel(self, filename: str, layout: DataLayoutSettings) -> SampleDataset:
        sheet = open_workbook(filename).sheet_by_index(0)
        raw_data = []
        for i in range(sheet.nrows):
            raw_data.append(sheet.row_values(i))
        classes, names, distributions = self.process_raw_data(raw_data, layout)
        dataset = SampleDataset()
        dataset.add_batch(classes, names, distributions)
        return dataset
