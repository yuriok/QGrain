

import logging
import os
from typing import List

import numpy as np
from PySide2.QtCore import QObject, Qt, QThread, Signal
from PySide2.QtWidgets import QFileDialog

from data import DataLoader, DataWriter, FittedData, GrainSizeData, SampleData
from resolvers import DistributionType


class DataManager(QObject):
    sigDataLoadingStarted = Signal(str, str)  # filename, file type
    sigDataLoadingFinished = Signal(bool) #TODO: CONNECT TO UI
    sigDataSavingStarted = Signal(str, np.ndarray, list, str)
    sigDataSavingFinished = Signal(bool) #TODO: CONNECT TO UI
    sigDataLoaded = Signal(GrainSizeData)
    sigTargetDataChanged = Signal(str, np.ndarray, np.ndarray)
    sigDataRecorded = Signal(FittedData)

    def __init__(self):
        super().__init__()
        self.grain_size_data = None  # type: GrainSizeData
        self.current_fitted_data = None  # type: FittedData
        self.recorded_data_list = []  # type: List[FittedData]
        self.file_dialog = QFileDialog()
        self.data_loader = DataLoader()
        self.load_data_thread = QThread()
        # move it before the signal-slots are connected
        self.data_loader.moveToThread(self.load_data_thread)
        self.load_data_thread.start()

        self.data_writer = DataWriter()
        self.save_data_thread = QThread()
        self.data_writer.moveToThread(self.save_data_thread)
        self.save_data_thread.start()

        self.sigDataLoadingStarted.connect(self.data_loader.try_load_data)
        self.data_loader.sigWorkFinished.connect(self.on_loading_work_finished)
        self.sigDataSavingStarted.connect(self.data_writer.try_save_data)
        self.data_writer.sigWorkFinished.connect(self.on_saving_work_finished)

        self.auto_record = True

    def load_data(self):
        filename, type_str = self.file_dialog.getOpenFileName(None, self.tr("Select Data File"), None, "*.xls; *.xlsx;;*.csv")
        if filename is None or filename == "":
            return
        if not os.path.exists(filename):
            return
        if ".xls" in type_str:
            file_type = "excel"
        elif ".csv" in type_str:
            file_type = "csv"
        else:
            raise ValueError(type_str)
        logging.info("Selected data file is [%s].", filename)
        self.sigDataLoadingStarted.emit(filename, file_type)

    def on_loading_work_finished(self, grain_size_data: GrainSizeData):
        if grain_size_data.is_valid:
            self.grain_size_data = grain_size_data
            self.sigDataLoadingFinished.emit(True)
            self.sigDataLoaded.emit(grain_size_data)
        else:
            self.sigDataLoadingFinished.emit(True)

    def on_focus_sample_changed(self, index: int):
        if self.grain_size_data is None:
            return
        sample_name = self.grain_size_data.sample_data_list[index].name
        classes = self.grain_size_data.classes
        sample_data = self.grain_size_data.sample_data_list[index].distribution

        self.sigTargetDataChanged.emit(sample_name, classes, sample_data)

    # TODO: ADD DATA VALIDATATION
    def on_epoch_finished(self, data: FittedData):
        print("Statistic for {0}:\n".format(data.name) +
              "|{0:12}|{1:12}|{2:12}|{3:12}|{4:12}|{5:12}|{6:24}|{7:12}|{8:12}|\n".format(
                  "Component", "Fraction", "Mean", "Median", "Mode", "Variance", "Standard Deviation", "Skewness", "Kurtosis") +
              "\n".join(["|{0:12}|{1:<12.2f}|{2:<12.2f}|{3:<12.2f}|{4:<12.2f}|{5:<12.2f}|{6:<24.2f}|{7:<12.2f}|{8:<12.2f}|".format(
                  i.get("name"), i.get("fraction"), i.get("mean"), i.get("median"), i.get("mode"), i.get("variance"), i.get("standard_deviation"), i.get("skewness"), i.get("kurtosis")) for i in data.statistic]))
        print("Mean Squared Error:", data.mse)
        self.current_fitted_data = data

        if self.auto_record:
            self.record_data()

    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            self.__setattr__(setting, value)

    def record_data(self):
        self.recorded_data_list.append(self.current_fitted_data)
        self.sigDataRecorded.emit(self.current_fitted_data)

    def remove_data(self, rows: List[int]):
        print(rows)
        offset = 0
        for row in rows:
            value_to_remove = self.recorded_data_list[row-offset]
            logging.info("Record of {0} has been removed.".format(value_to_remove.name))
            self.recorded_data_list.remove(value_to_remove)
            offset += 1

    def save_data(self):
        filename, type_str = self.file_dialog.getSaveFileName(None, self.tr("Save Recorded Data"), None, "Excel Workbook (*.xlsx);;97-2003 Excel Workbook (*.xls);;CSV (*.csv)")
        logging.info(self.tr("File path is [{0}].").format(filename))
        if filename is None or filename == "":
            return
        if ".xlsx" in type_str:
            file_type = "xlsx"
        elif "97-2003" in type_str:
            file_type = "xls"
        elif ".csv" in type_str:
            file_type = "csv"
        else:
            raise ValueError(type_str)
        
        self.sigDataSavingStarted.emit(filename, self.grain_size_data.classes, self.recorded_data_list, file_type)

    def on_saving_work_finished(self, state):
        if state:
            logging.info("File saved.")
        else:
            logging.error("File unsaved.")
