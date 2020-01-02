

import logging
import os
from typing import Iterable, List, Tuple, Callable
from uuid import UUID

import numpy as np
from PySide2.QtCore import QObject, QStandardPaths, Qt, Signal, QThread
from PySide2.QtWidgets import QFileDialog, QMessageBox, QWidget, QProgressDialog

from algorithms import DistributionType
from models.FittingResult import *
from models.DataLoader import *
from models.DataWriter import *
from models.DataLayoutSetting import *
from models.SampleData import *
from models.SampleDataset import *


from resolvers import FittingTask


class BackgroundLoader(QObject):
    sigWorkSucceeded = Signal(SampleDataset)
    sigWorkFailed = Signal(Exception)

    def __init__(self):
        super().__init__()
        self.actual_loader = DataLoader()

    def on_work_started(self, filename: str, file_type: FileType,
                        layout: DataLayoutSetting):
        try:
            dataset = self.actual_loader.try_load_data(filename, file_type, layout)
            self.sigWorkSucceeded.emit(dataset)
        except Exception as e:
            self.sigWorkFailed.emit(e)


class BackgroundWriter(QObject):
    sigWorkSucceeded = Signal()
    sigWorkFailed = Signal(Exception)

    def __init__(self):
        super().__init__()
        self.actual_writer = DataWriter()

    def on_work_started(self, filename: str, file_type: FileType,
                        results: List[FittingResult]):
        try:
            dataset = self.actual_writer.try_save_data(filename, file_type, results)
            self.sigWorkSucceeded.emit()
        except Exception as e:
            self.sigWorkFailed.emit(e)


class DataManager(QObject):
    sigLoadingStarted = Signal(str, FileType, DataLayoutSetting)
    sigSavingStarted = Signal(str, FileType, list)
    sigDataLoaded = Signal(SampleDataset)
    sigTargetDataChanged = Signal(SampleData)
    sigDataRecorded = Signal(list) # List[FittingResult]
    logger = logging.getLogger("root.data.DataManager")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, host_widget: QWidget):
        super().__init__()
        # to attach msg boxed on this widget
        self.host_widget = host_widget
        # data
        self.dataset = None  # type: SampleDataset
        self.current_fitting_result = None  # type: FittingResult
        self.records = []  # type: List[FittingResult]
        
        # loader
        self.loader = BackgroundLoader()
        self.loading_thread = QThread()
        self.loader.moveToThread(self.loading_thread)
        self.sigLoadingStarted.connect(self.loader.on_work_started)
        self.loader.sigWorkSucceeded.connect(self.on_loading_succeeded)
        self.loader.sigWorkFailed.connect(self.on_loading_failed)

        # writer
        self.writer = BackgroundWriter()
        self.saving_thread = QThread()
        self.writer.moveToThread(self.saving_thread)
        self.sigSavingStarted.connect(self.writer.on_work_started)
        self.writer.sigWorkSucceeded.connect(self.on_saving_succeeded)
        self.writer.sigWorkFailed.connect(self.on_saving_failed)

        # settings
        self.data_layout_setting = DataLayoutSetting()
        self.auto_record_flag = True

        self.file_dialog = QFileDialog(self.host_widget)
        self.msg_box = QMessageBox(self.host_widget)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.retry_msg_box = QMessageBox(self.host_widget)
        self.retry_msg_box.addButton(QMessageBox.StandardButton.Retry)
        self.retry_msg_box.addButton(QMessageBox.StandardButton.Ok)
        self.retry_msg_box.setDefaultButton(QMessageBox.StandardButton.Retry)
        self.retry_msg_box.setWindowFlags(Qt.Drawer)

        self.record_msg_box = QMessageBox(self.host_widget)
        self.record_msg_box.addButton(QMessageBox.StandardButton.Discard)
        self.record_msg_box.addButton(QMessageBox.StandardButton.Ok)
        self.record_msg_box.setDefaultButton(QMessageBox.StandardButton.Discard)
        self.record_msg_box.setWindowFlags(Qt.Drawer)


    def show_message(self, title: str, message: str):
        self.msg_box.setWindowTitle(title)
        self.msg_box.setText(message)
        self.msg_box.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    def ask_retry(self, message: str, func: Callable):
        self.retry_msg_box.setWindowTitle(self.tr("Error"))
        self.retry_msg_box.setText(message)
        result = self.retry_msg_box.exec_()
        if result == QMessageBox.Retry:
            func()

    def load_data(self):
        # NOTE:
        # if don't assign the initial directory,
        # there is an about 5s delay while first calling `getOpenFileName` func
        # if there is a unreachable network disk
        init_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
        filename, type_str = self.file_dialog.getOpenFileName(self.host_widget, self.tr("Select Data File"), init_path, "Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)")
        if filename is None or filename == "":
            self.logger.info("The user did not select a file, ignored.")
            return
        # get the file type info
        if ".xlsx" in type_str:
            file_type = FileType.XLSX
        elif "97-2003" in type_str:
            file_type = FileType.XLS
        elif ".csv" in type_str:
            file_type = FileType.CSV
        else:
            raise NotImplementedError(type_str)

        self.logger.info("Try to load data, selected data file is [%s].", filename)
        self.sigLoadingStarted.emit(filename, file_type, self.data_layout_setting)

    def on_loading_succeeded(self, dataset: SampleDataset):
        if self.dataset is None:
            self.dataset = dataset
        else:
            try:
                self.dataset.combine(dataset)
            except ClassesNotMatchError:
                self.logger.exception("The new batch of data's classes are not equal to the existing's.", stack_info=True)
                self.ask_retry(self.tr("The new batch of data's classes are not equal to the existing's."), self.load_data)
                return
        self.sigDataLoaded.emit(self.dataset)
        self.logger.info("Data has been loaded.")
        self.show_info(self.tr("Data has been loaded."))

    def on_loading_failed(self, exception: Exception):
        try:
            raise exception
        # handle the exceptions
        except PermissionError:
            self.logger.exception("Can not access the selected file due to permission issue. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("Can not access the selected file due to permission issue, please check whether it's opened by another program."), self.load_data)
        except FileNotFoundError:
            self.logger.exception("The selected file does not exist. Maybe it's removed by others. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("The selected file does not exist now, please check whether it's removed."), self.load_data)
        except XLRDError:
            self.logger.exception("A exception raised while reading Excel file. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("Can not read data from this Excel file."), self.load_data)
        except CSVEncodingError:
            self.logger.exception("The encoding of this CSV file is not utf-8. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("Please make sure the encoding of CSV file is utf-8."), self.load_data)
        except ValueNotNumberError:
            self.logger.exception("Can not convert the value to real number. Data layout setting: [%s].", self.data_layout_setting, stack_info=True)
            self.ask_retry(self.tr("The value can not be converted to real number.\nPlease make sure the data layout setting matchs your data file."), self.load_data)
        except ClassesNotIncrementalError:
            self.logger.exception("The classes values are not incremental. Filename: [%s]. Data layout setting: [%s].", filename, self.data_layout_setting, stack_info=True)
            self.ask_retry(self.tr("The array of grain size classes is not incremental.\nPlease make sure the data file is correct and the data layout setting matchs the file."), self.load_data)
        except ClassesNotMatchError:
            self.logger.exception("The new batch of data's classes are not equal to the existing's.", stack_info=True)
            self.ask_retry(self.tr("The new batch of data's classes are not equal to the existing's."), self.load_data)
        except NaNError:
            self.logger.exception("There is a NaN value in this data file.", stack_info=True)
            self.ask_retry(self.tr("There is a NaN value in this data file."), self.load_data)
        except ArrayEmptyError:
            self.logger.exception("There is an array that is empty.", stack_info=True)
            self.ask_retry(self.tr("There is an array that is empty.\nIf you are sure that the CSV file and layout setting are correct, please check the dialect of CSV is excel, NOT excel-tab."), self.load_data)
        except SampleNameEmptyError:
            self.logger.exception("At leaset one sample name is empty.", stack_info=True)
            self.ask_retry(self.tr("At leaset one sample name is empty. Please give each sample a unique name to identify."), self.load_data)
        except DistributionSumError:
            self.logger.exception("The sum of distribution array is not equal to 1 or 100.", stack_info=True)
            self.ask_retry(self.tr("The sum of distribution array is not equal to 1 or 100. Please make sure the data file you selected is grain size distribution data"), self.load_data)
        except Exception:
            self.logger.exception("Unknown exception raised.", stack_info=True)
            self.ask_retry(self.tr("Unknown exception raised. See the log for more details."), self.load_data)

    def on_focus_sample_changed(self, index: int):
        if self.dataset is None:
            self.logger.info("Grain size data is still None, ignored.")
            return
        self.sigTargetDataChanged.emit(self.dataset.samples[index])
        self.logger.debug("Focus sample data changed, the data has been emitted.")

    def on_fitting_epoch_suceeded(self, result: FittingResult):
        self.logger.info("Epoch for sample [%s] has finished, mean squared error is [%E].", result.name, result.mean_squared_error)
        self.current_fitting_result = result
        if self.auto_record_flag:
            if result.has_invalid_value:
                self.record_msg_box.setWindowTitle(self.tr("Warning"))
                self.record_msg_box.setText(self.tr("The fitted data may be invalid, at least one NaN value occurs."))
                exec_result = self.record_msg_box.exec_()
                if exec_result == QMessageBox.Discard:
                    self.logger.info("Fitted data of sample [%s] was discarded by user.", result.name)
                    return
                else:
                    self.record_current_data()
            else:
                self.record_current_data()

    def on_multiprocessing_task_finished(self, succeeded_results: Iterable[FittingResult], failed_tasks: Iterable[FittingTask]):
        for fitting_result in succeeded_results:
            if fitting_result.has_invalid_value:
                self.logger.warning("There is invalid value in the fitted data of sample [%s].", fitting_result.name)
                self.gui_logger.warning(self.tr("There is invalid value in the fitted data of sample [%s]."), fitting_result.name)
        self.records.extend(succeeded_results)
        self.sigDataRecorded.emit(succeeded_results)
        for failed_task in failed_tasks:
            self.logger.warning("Fitting task of sample [%s] failed.", failed_task.sample_name)
            self.gui_logger.warning(self.tr("Fitting task of sample [%s] failed."), failed_task.sample_name)

    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            setattr(self, setting, value)
            self.logger.info("Setting [%s] have been changed to [%s].", setting, value)

    def record_current_data(self):
        if self.current_fitting_result is None:
            self.logger.info("There is no fitted data to record, ignored.")
            self.gui_logger.warning(self.tr("There is no fitted data to record."))
            self.msg_box.setWindowTitle(self.tr("Warning"))
            self.msg_box.setText(self.tr("There is no fitted data to record."))
            self.msg_box.exec_()
            return
        self.records.append(self.current_fitting_result)
        self.sigDataRecorded.emit([self.current_fitting_result])

    def remove_data(self, uuids_and_names: Iterable[Tuple[UUID, str]]):
        for uuid, name in uuids_and_names:
            for i, data in enumerate(self.records):
                if uuid == data.uuid:
                    assert name == data.name
                    self.records.pop(i)
                    self.logger.info("Record of sample [%s] has been removed.", name)
                    break

    def save_data(self):
        filename, type_str = self.file_dialog.getSaveFileName(None, self.tr("Save Recorded Data"), None, "Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)")
        self.logger.info("File path to save is [%s].", filename)
        if filename is None or filename == "":
            self.logger.info("The path is None or empty, ignored.")
            return
        if os.path.exists(filename):
            self.logger.warning("This file has existed and will be replaced. Filename: %s.", filename)

        if ".xlsx" in type_str:
            file_type = FileType.XLSX
        elif "97-2003" in type_str:
            file_type = FileType.XLS
        elif ".csv" in type_str:
            file_type = FileType.CSV
        else:
            raise NotImplementedError(type_str)
        self.logger.info("Selected file type is [%s].", file_type)
        self.sigSavingStarted.emit(filename, file_type, self.records)

    def on_saving_succeeded(self):
            self.logger.info("File saved.")
            self.show_info(self.tr("The data has been saved to the file."))

    def on_saving_failed(self, exception: Exception):
        try:
            raise exception
        except PermissionError:
            self.logger.exception("Can not access the selected file due to permission issue. Filename: [%s].", filename, stack_info=True)
            self.ask_retry(self.tr("Can not access the selected file due to permission issue, please check whether it's opened by another program."), self.save_data)
        except Exception:
            self.logger.exception("Unknown exception raised.", stack_info=True)
            self.ask_retry(self.tr("Unknown exception raised. See the log for more details."), self.save_data)

    def setup_all(self):
        self.loading_thread.start()
        self.saving_thread.start()

    def cleanup_all(self):
        self.loading_thread.terminate()
        self.saving_thread.terminate()
