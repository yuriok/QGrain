import csv
import logging

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from xlrd import open_workbook
from xlrd.biffh import XLRDError

from data import GrainSizeData, SampleData
from typing import List, Iterable

class DataInvalidError(Exception):
    def __ini__(self):
        super().__init__()


class DataFormatSetting:
    def __init__(self):
        self.classes_row = 0
        self.sample_name_column = 0
        self.data_start_row = 1
        self.data_start_column = 1

class DataLoader(QObject):
    sigWorkFinished = pyqtSignal(GrainSizeData)
    logger = logging.getLogger("root.data.DataLoader")
    
    def __init__(self):
        super().__init__()
        self.setting = DataFormatSetting()

    def try_load_data(self, filename, file_type):
        if file_type == "excel":
            self.try_excel(filename)
        elif file_type == "csv":
            self.try_csv(filename)

    def try_excel(self, filename):
        try:
            sheet = open_workbook(filename).sheet_by_index(0)
        except XLRDError:
            self.logger.warning("The file format is not excel. Filename: [%s].", filename)
            self.sigWorkFinished.emit(GrainSizeData())
            return
        try:
            raw_data = []
            for i in range(sheet.nrows):
                raw_data.append(sheet.row_values(i))
        except Exception:
            self.logger.exception("Unknown exception raised during reading the data table. Filename: [%s].", filename, stack_info=True)
            self.sigWorkFinished.emit(GrainSizeData())
            return
        try:
            grain_size_data = self.process_raw_data(raw_data, self.setting)
            self.logger.info("Data has been loaded from the excel file. Filename: [%s].", filename)
            self.sigWorkFinished.emit(grain_size_data)
            return
        except DataInvalidError:
            self.logger.exception("The data in it is not valid. Filename: [%s].", filename, stack_info=True)
            self.sigWorkFinished.emit(GrainSizeData())
            return
        except Exception:
            self.logger.exception("Unknown exception raised. Maybe the data format is not qualified. Filename: [%s].", filename, stack_info=True)
            self.sigWorkFinished.emit(GrainSizeData())
            return

    def try_csv(self, filename):
        try:
            # open file
            f = open(filename, encoding="utf-8")
        except Exception:
            self.logger.exception("Can not open the file. Filename: [%s].", filename, stack_info=True)
            self.sigWorkFinished.emit(GrainSizeData())
            return
        try:
            # read
            r = csv.reader(f)
            raw_data = [row for row in r]
        except Exception:
            self.logger.exception("Can not read the file as csv. Check the encode and make sure it's [utf-8]. Filename: [%s].", filename, stack_info=True)
            self.sigWorkFinished.emit(GrainSizeData())
            return
        try:
            grain_size_data = self.process_raw_data(raw_data, self.setting)
            self.logger.info("Grain size data has been loaded from the csv file. Filename: [%s].", filename)
            self.sigWorkFinished.emit(grain_size_data)
            return
        except DataInvalidError:
            self.logger.exception("The data in it is not valid. Filename: [%s].", filename, stack_info=True)
            self.sigWorkFinished.emit(GrainSizeData())
            return
        except Exception:
            self.logger.exception("Unknown exception raised. Maybe the data format is not qualified. Filename: [%s].", filename, stack_info=True)
            self.sigWorkFinished.emit(GrainSizeData())
            return


    def process_raw_data(self, raw_data: List[List], setting: DataFormatSetting):
        # convert data
        classes = np.array(raw_data[setting.classes_row][setting.data_start_column:], dtype=float)
        sample_data_list = []
        for row_values in raw_data[setting.data_start_row:]:
            sample_data_list.append(SampleData(row_values[setting.sample_name_column], np.array(
                row_values[setting.data_start_column:], dtype=float)))
        
        self.validate_data(classes, sample_data_list)
        grain_size_data = GrainSizeData(is_valid=True, classes=classes, sample_data_list=sample_data_list)
        return grain_size_data


    def is_incremental(self, nums: Iterable):
        for i in range(1, len(nums)):
            if nums[i] <= nums[i-1]:
                return False
        return True

    def validate_data(self, classes: np.ndarray, sample_data_list: List[SampleData]):
        # the classes must be incremental
        if not self.is_incremental(classes):
            raise DataInvalidError("The values of classes are not incremental. [{0}]".format(classes))
        single_data_length = len(classes)
        # each data must has same length as classes
        sum_equals_hundred = []
        for sample_data in sample_data_list:
            if sample_data.name is None or sample_data == "":
                raise DataInvalidError("At least one sample's name is empty.")
            if len(sample_data.distribution) != single_data_length:
                raise DataInvalidError("The length of sample data [{0}] is not equal to that of classes.".format(sample_data.name))
            # check the sum is close to 1 or 100 (someone may use percentage)
            s = np.sum(sample_data.distribution)
            if s > 0.99 and s < 1.01:
                sum_equals_hundred.append(False)
            elif s > 99 and s < 101:
                sum_equals_hundred.append(True)
            else:
                raise DataInvalidError("The sum of data [{0}] is not equal to 1 or 100.".format(sample_data.name))
        state0 = sum_equals_hundred[0]
        for state in sum_equals_hundred[1:]:
            if state != state0:
                raise DataInvalidError("Not all sum values equals to the same value.")

