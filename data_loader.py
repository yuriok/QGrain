import csv
import logging
import os

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from xlrd import open_workbook
from xlrd.biffh import XLRDError

from data import GrainSizeData, SampleData


class DataLoader(QObject):
    sigWorkFinished = pyqtSignal(GrainSizeData)

    CLASSES_ROW = 0
    SAMPLE_NAME_COLUMN = 0
    DATA_START_ROW = 1
    DATA_START_COLUMN = 1

    def __init__(self):
        super().__init__()
        self.last_loaded_filename = None

    def try_load_data(self, filename):
        if filename is None or filename == "":
            raise ValueError(filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        try:
            self.try_excel(filename)
        except XLRDError:
            logging.warning(
                self.tr("The format is not excel, try csv.\nFilename: {0}.").format(filename))
            try:
                self.try_csv(filename)
            except Exception:
                logging.exception(self.tr("Unknown exception raised.\nFilename: {0}.").format(
                    filename), stack_info=True)
                self.sigWorkFinished.emit(None)

    def try_excel(self, filename):
        sheet = open_workbook(filename).sheet_by_index(self.CLASSES_ROW)
        classes = np.array(sheet.row_values(self.CLASSES_ROW)[
                           self.DATA_START_COLUMN:], dtype=float)
        sample_data_list = []
        for i in range(self.DATA_START_ROW, sheet.nrows):
            row_values = sheet.row_values(i)
            sample_data_list.append(SampleData(row_values[self.SAMPLE_NAME_COLUMN], np.array(
                row_values[self.DATA_START_COLUMN:], dtype=float)))

        grain_size_data = GrainSizeData(classes, sample_data_list)
        logging.info(self.tr(
            "Grain size data has been loaded from the excel file.\nFilename: {0}.").format(filename))
        self.sigWorkFinished.emit(grain_size_data)
        self.last_loaded_filename = filename

    def try_csv(self, filename):
        with open(filename, encoding="utf-8") as f:
            r = csv.reader(f)
            for _ in range(self.CLASSES_ROW-1):
                next(r)
            classes = np.array(next(r)[self.DATA_START_COLUMN:], dtype=float)
            for _ in range(self.DATA_START_ROW-self.CLASSES_ROW-1):
                next(r)
            sample_data_list = []
            for row_values in r:
                sample_data_list.append(SampleData(row_values[self.SAMPLE_NAME_COLUMN], np.array(
                    row_values[self.DATA_START_COLUMN:], dtype=float)))

            grain_size_data = GrainSizeData(classes, sample_data_list)
            logging.info(self.tr(
                "Grain size data has been loaded from the csv file.\nFilename: {0}.").format(filename))
            self.sigWorkFinished.emit(grain_size_data)
            self.last_loaded_filename = filename
