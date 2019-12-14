

import logging
import os
from typing import List

import numpy as np
from PySide2.QtCore import QObject, Qt, QThread, Signal
from PySide2.QtWidgets import QFileDialog, QMessageBox

from algorithms import DistributionType
from data import DataLoader, DataWriter, FittedData, GrainSizeData, SampleData
from resolvers import FittingTask


class DataManager(QObject):
    sigDataLoadingStarted = Signal(str, str)  # filename, file type
    sigDataSavingStarted = Signal(str, np.ndarray, list, str)
    sigDataLoaded = Signal(GrainSizeData)
    sigTargetDataChanged = Signal(str, np.ndarray, np.ndarray)
    sigDataRecorded = Signal(FittedData)
    logger = logging.getLogger("root.data.DataManager")
    gui_logger = logging.getLogger("GUI")

    def __init__(self):
        super().__init__()
        # Data
        self.grain_size_data = None  # type: GrainSizeData
        self.current_fitted_data = None  # type: FittedData
        self.recorded_data_list = []  # type: List[FittedData]
        
        # Loader
        self.data_loader = DataLoader()
        self.load_data_thread = QThread()
        # move it before the signal-slots are connected
        self.data_loader.moveToThread(self.load_data_thread)
        self.load_data_thread.start()
        # Writer
        self.data_writer = DataWriter()
        self.save_data_thread = QThread()
        self.data_writer.moveToThread(self.save_data_thread)
        self.save_data_thread.start()

        self.sigDataLoadingStarted.connect(self.data_loader.try_load_data)
        self.data_loader.sigWorkFinished.connect(self.on_loading_work_finished)
        self.sigDataSavingStarted.connect(self.data_writer.try_save_data)
        self.data_writer.sigWorkFinished.connect(self.on_saving_work_finished)

        # Settings
        self.auto_record_flag = True

        self.file_dialog = QFileDialog()
        self.msg_box = QMessageBox()
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.load_msg_box = QMessageBox()
        self.load_msg_box.addButton(QMessageBox.StandardButton.Retry)
        self.load_msg_box.addButton(QMessageBox.StandardButton.Ok)
        self.load_msg_box.setDefaultButton(QMessageBox.StandardButton.Retry)
        self.load_msg_box.setWindowFlags(Qt.Drawer)
        self.record_msg_box = QMessageBox()
        self.record_msg_box.addButton(QMessageBox.StandardButton.Discard)
        self.record_msg_box.addButton(QMessageBox.StandardButton.Ok)
        self.record_msg_box.setDefaultButton(QMessageBox.StandardButton.Discard)
        self.record_msg_box.setWindowFlags(Qt.Drawer)

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
        self.logger.info("Try to load data, selected data file is [%s].", filename)
        self.sigDataLoadingStarted.emit(filename, file_type)

    def on_loading_work_finished(self, grain_size_data: GrainSizeData):
        if grain_size_data.is_valid:
            self.grain_size_data = grain_size_data
            self.sigDataLoaded.emit(grain_size_data)
            self.logger.debug("Data has been loaded.")
            self.gui_logger.info(self.tr("Data has been loaded."))
            self.msg_box.setWindowTitle(self.tr("Info"))
            self.msg_box.setText(self.tr("The data has been loaded from the file."))
            self.msg_box.exec_()
        else:
            self.logger.warning("Data has not been loaded correctly.")
            self.gui_logger.error(self.tr("Data has not been loaded correctly, check and try it again please."))
            self.load_msg_box.setWindowTitle(self.tr("Error"))
            self.load_msg_box.setText(self.tr("Data loading failed."))
            result = self.load_msg_box.exec_()
            if result == QMessageBox.Retry:
                self.load_data()

    def on_focus_sample_changed(self, index: int):
        if self.grain_size_data is None:
            self.logger.info("Grain size data is still None, ignored.")
            # TODO: add msg box to hint that whether to load data
            return
        sample_name = self.grain_size_data.sample_data_list[index].name
        classes = self.grain_size_data.classes
        sample_data = self.grain_size_data.sample_data_list[index].distribution
        self.sigTargetDataChanged.emit(sample_name, classes, sample_data)
        self.logger.debug("Focus sample data changed, the data has been emitted.")

    def on_fitting_epoch_suceeded(self, data: FittedData):
        self.logger.info("Epoch for sample [%s] has finished, mean squared error is [%E], statistic is: [%s].", data.name, data.mse, data.statistic)
        self.current_fitted_data = data
        if self.auto_record_flag:
            if data.has_invalid_value():
                self.record_msg_box.setWindowTitle(self.tr("Warning"))
                self.record_msg_box.setText(self.tr("The fitted data may be invalid, at least one NaN value occurs."))
                result = self.record_msg_box.exec_()
                if result == QMessageBox.Discard:
                    self.logger.info("Fitted data of sample [%s] was discarded by user.", data.name)
                    return
                else:
                    self.record_current_data()
            else:
                self.record_current_data()

    def on_multiprocessing_task_finished(self, succeeded_results: List[FittedData], failed_tasks: List[FittingTask]):
        for fitted_data in succeeded_results:
            if fitted_data.has_invalid_value():
                self.logger.warning("There is invalid value in the fitted data of sample [%s].", fitted_data.name)
                self.gui_logger.warning(self.tr("There is invalid value in the fitted data of sample [%s]."), fitted_data.name)
            self.recorded_data_list.append(fitted_data)
            self.sigDataRecorded.emit(fitted_data)
        for failed_task in failed_tasks:
            self.logger.warning("Fitting task of sample [%s] failed.", failed_task.sample_name)
            self.gui_logger.warning(self.tr("Fitting task of sample [%s] failed."), failed_task.sample_name)

    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            setattr(self, setting, value)
            self.logger.info("Setting [%s] have been changed to [%s].", setting, value)

    def record_current_data(self):
        if self.current_fitted_data is None:
            self.logger.info("There is no fitted data to record, ignored.")
            self.gui_logger.warning(self.tr("There is no fitted data to record."))
            self.msg_box.setWindowTitle(self.tr("Warning"))
            self.msg_box.setText(self.tr("There is no fitted data to record."))
            self.msg_box.exec_()
            return
        self.recorded_data_list.append(self.current_fitted_data)
        self.sigDataRecorded.emit(self.current_fitted_data)

    def remove_data(self, rows: List[int]):
        offset = 0
        for row in rows:
            value_to_remove = self.recorded_data_list[row-offset]
            self.recorded_data_list.remove(value_to_remove)
            self.logger.info("Record of sample [%s] has been removed.", value_to_remove.name)
            offset += 1

    def save_data(self):
        filename, type_str = self.file_dialog.getSaveFileName(None, self.tr("Save Recorded Data"), None, "Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)")
        self.logger.info("File path to save is [%s].", filename)
        if filename is None or filename == "":
            self.logger.info("The path is None or empty, ignored.")
            return
        if ".xlsx" in type_str:
            file_type = "xlsx"
        elif "97-2003" in type_str:
            file_type = "xls"
        elif ".csv" in type_str:
            file_type = "csv"
        else:
            raise ValueError(type_str)
        self.logger.info("Selected file type is [%s].", file_type)
        self.sigDataSavingStarted.emit(filename, self.grain_size_data.classes, self.recorded_data_list, file_type)

    def on_saving_work_finished(self, state):
        if state:
            self.logger.info("File saved.")
            self.gui_logger.info(self.tr("File saved."))
            self.msg_box.setWindowTitle("Info")
            self.msg_box.setText(self.tr("The data has been saved to the file."))
            self.msg_box.exec_()
        else:
            self.logger.error("File unsaved.")
            self.msg_box.setWindowTitle(self.tr("Error"))
            self.msg_box.setText(self.tr("Data saving failed."))
            self.msg_box.exec_()
