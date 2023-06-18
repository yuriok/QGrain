import csv
import logging
import os
from enum import Enum, unique

import numpy as np
import openpyxl
import xlrd
from typing import *

from ..models.dataset import Dataset, validate_classes, validate_distributions


def check_layout(class_row: int, name_col: int, start_row: int, start_col: int):
    assert isinstance(class_row, int)
    assert isinstance(name_col, int)
    assert isinstance(start_row, int)
    assert isinstance(start_col, int)
    if class_row < 0 or \
            name_col < 0 or \
            start_row < 0 or \
            start_col < 0:
        raise ValueError("The index of row or column must be non-negative.")
    if class_row >= start_row:
        raise ValueError("The start row index of distributions must be greater than the row index of classes.")
    if name_col >= start_col:
        raise ValueError("The start column index of distributions must be greater "
                         "than the column index of sample names.")


@unique
class ReadFileType(Enum):
    XLS = 0
    XLSX = 1
    CSV = 2


def get_file_type(filename: str):
    _, extension = os.path.splitext(filename)
    if extension == ".csv":
        return ReadFileType.CSV
    elif extension == ".xls":
        return ReadFileType.XLS
    elif extension == ".xlsx":
        return ReadFileType.XLSX
    else:
        raise NotImplementedError(extension)


def _get_raw_table(
        filename: str, file_type: ReadFileType, sheet_index: int,
        progress_callback: Callable[[float], None] = None,
        logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger("QGrain")
    else:
        assert isinstance(logger, logging.Logger)
    logger.debug(f"Try to load the raw data table from [{file_type.name}] file: {filename}.")
    if file_type == ReadFileType.CSV:
        with open(filename, encoding="utf-8") as f:
            reader = csv.reader(f)
            raw_table = []
            for i, row_values in enumerate(reader):
                raw_table.append(row_values)
                if progress_callback is not None:
                    progress_callback(i / reader.line_num)

    elif file_type == ReadFileType.XLS:
        workbook: xlrd.Book = xlrd.open_workbook(filename)
        sheet = workbook.sheet_by_index(sheet_index)
        raw_table = []
        for row in range(sheet.nrows):
            raw_table.append(sheet.row_values(row))
            if progress_callback is not None:
                progress_callback(row / sheet.nrows)

    elif file_type == ReadFileType.XLSX:
        workbook: openpyxl.Workbook = openpyxl.load_workbook(filename, read_only=True, data_only=True)
        sheet = workbook[workbook.sheetnames[sheet_index]]
        raw_table = []
        for i, row_values in enumerate(sheet.values):
            raw_table.append(row_values)
            if progress_callback is not None:
                progress_callback(i / sheet.max_row)
    else:
        raise NotImplementedError(file_type)
    logger.debug("The raw data table has been loaded from this file.")
    return raw_table


def load_dataset(filename: str,
                 dataset_name: str = None,
                 sheet_index: int = 0,
                 class_row: int = 0,
                 name_col: int = 0,
                 start_row: int = 1,
                 start_col: int = 1,
                 skip_invalid_rows=True,
                 progress_callback: Callable[[float], None] = None,
                 logger: logging.Logger = None) -> Union[Dataset, None]:
    """
    Try to load the grain size dataset from a file.

    A data files of any types can be regarded as the table that contains rows and columns.

    For the file which contains grain size distributions, it should be the following format:
        * The first valid row should be the headers (i.e. the grain size classes in microns).
        * The following valid rows should be the distributions of samples corresponding to the grain size classes.
        * The first valid column should be the names of samples.
    To make it more flexible, you can use the settings to control the data loader.

    :param filename: The file path which contains the extension.
    :param dataset_name: The name of this dataset. If not indicate it, it will use the filename.
    :param sheet_index: If it is an Excel file (`*.xls` or `*.xlsx`), please indicate the sheet index,
        otherwise it will use the first sheet.
    :param class_row: The row index of the grain size classes.
    :param name_col: The column index of the sample names.
    :param start_row: The start row index of the grain size distributions.
    :param start_col: The start column index of the grain size distributions.
    :param skip_invalid_rows: If `True`, it will skip the invalid rows, rather than break up.
    :param progress_callback: The callable object which will be called at each step to report the progress.
    :param logger: The logger which will be used to log the information of loading.
    :return: If there is any exception raised, it will return `None`, else return a `Dataset` object.
    """
    assert isinstance(filename, str)
    assert len(filename) != 0
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(filename))[0]
    else:
        assert isinstance(dataset_name, str)
        assert len(dataset_name) != 0
    assert isinstance(sheet_index, int)
    assert sheet_index >= 0
    check_layout(class_row, name_col, start_row, start_col)
    if logger is None:
        logger = logging.getLogger("QGrain")
    else:
        assert isinstance(logger, logging.Logger)

    file_type = get_file_type(filename)
    try:
        raw_table = _get_raw_table(
            filename, file_type, sheet_index,
            progress_callback=None if progress_callback is None else lambda p: progress_callback(p * 0.5))
        class_values = raw_table[class_row][start_col:]
        valid, array_or_msg = validate_classes(class_values)
        if valid:
            classes = array_or_msg
            logger.debug(f"Grain size classes in μm: [{','.join([f'{x: 0.4f}' for x in classes])}].")
        else:
            logger.error(f"The assigned series of grain size classes is invalid. {array_or_msg}")
            return None
        sample_names = []
        distributions = []
        for row, row_values in enumerate(raw_table[start_row:], start_row + 1):
            # check if it's an empty row, i.e. the values all are empty string
            is_empty_row = True
            for frequency in row_values[start_col:]:
                if frequency is not None and frequency != "":
                    is_empty_row = False
                    break
            # if it's an empty row, jump this row to process the next one
            if is_empty_row:
                logger.warning(f"Row {row} is empty, skip to next row.")
                continue

            sample_name = row_values[name_col]
            if sample_name is None:
                sample_name = "NONE"
                logger.warning(f"The sample name at row {row} is `None`, use 'NONE' instead.")
            elif not isinstance(sample_name, str):
                sample_name = str(sample_name)
                logger.warning(f"The sample name at row {row} is not `str`, use `str` to covert it.")
            elif len(sample_name) == 0:
                sample_name = "EMPTY"
                logger.warning(f"The sample name at row {row} is empty, use `EMPTY` to covert it.")

            try:
                distribution = np.array(row_values[start_col:], dtype=np.float32)
            except ValueError as e:
                logger.error(f"Can not convert the frequencies at row {row} to a numerical array, "
                             f"it may contains invalid values (e.g. text or empty cell). {e}")
                if skip_invalid_rows:
                    continue
                else:
                    return None
            sample_names.append(sample_name)
            distributions.append(distribution)
            # logger.debug(f"The validation of sample [{sample_name}] at row [{current_row}] is passed.")
            if progress_callback is not None:
                progress = row / len(raw_table) * 0.5 + 0.5
                progress_callback(progress)

        valid, array_or_msg = validate_distributions(distributions)
        if valid:
            distributions = array_or_msg
            dataset = Dataset(dataset_name, sample_names, classes, distributions)
            if progress_callback is not None:
                progress_callback(1.0)
            logger.info("This dataset has been loaded successfully.")
            return dataset
        else:
            logger.error(f"There is at least one grain size distribution is invalid. {array_or_msg}")
    except IOError as e:
        logger.error(f"Can not open this file. {e}")
        return None
    except IndexError as e:
        logger.error(f"The row or column index is out of range, please check. {e}")
        return None
