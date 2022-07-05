__all__ = ["LoadDatasetDialog"]

import logging
import os
import typing

import openpyxl
import xlrd
from PySide6 import QtCore, QtWidgets

from ..io import *
from ..model import GrainSizeDataset


class LoadDatasetDialog(QtWidgets.QDialog):
    logger = logging.getLogger("QGrain.LoadDatasetDialog")
    dataset_loaded = QtCore.Signal(GrainSizeDataset)
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=QtCore.Qt.Window)
        self.setWindowTitle(self.tr("Dataset Loader"))
        self.initialize_ui()
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self.filename = None # type: str
        self.workbook = None # type: typing.Union[openpyxl.Workbook, xlrd.Book]
        self.dataset = None # type: GrainSizeDataset

    def initialize_ui(self):
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)

        self.filename_display = QtWidgets.QLabel(self.tr("Filename Unknown"))
        self.main_layout.addWidget(self.filename_display, 0, 0)
        self.select_button = QtWidgets.QPushButton(self.tr("Select"))
        self.select_button.clicked.connect(self.on_select_clicked)
        self.main_layout.addWidget(self.select_button, 0, 1)
        self.sheet_label = QtWidgets.QLabel(self.tr("Sheet Name"))
        self.sheet_combo_box = QtWidgets.QComboBox()
        self.sheet_combo_box.addItem(self.tr("Empty"))
        self.main_layout.addWidget(self.sheet_label, 1, 0)
        self.main_layout.addWidget(self.sheet_combo_box, 1, 1)

        self.classes_row_label = QtWidgets.QLabel(self.tr("Row With Grain Size Classes"))
        self.classes_row_input = QtWidgets.QSpinBox()
        self.classes_row_input.setRange(1, 999)
        self.main_layout.addWidget(self.classes_row_label, 2, 0)
        self.main_layout.addWidget(self.classes_row_input, 2, 1)
        self.sample_names_column_label = QtWidgets.QLabel(self.tr("Column With Sample Names"))
        self.sample_names_column_input = QtWidgets.QSpinBox()
        self.sample_names_column_input.setRange(1, 999)
        self.main_layout.addWidget(self.sample_names_column_label, 3, 0)
        self.main_layout.addWidget(self.sample_names_column_input, 3, 1)
        self.distribution_start_row_label = QtWidgets.QLabel(self.tr("Distribution Start Row"))
        self.distribution_start_row_input = QtWidgets.QSpinBox()
        self.distribution_start_row_input.setRange(2, 999999)
        self.main_layout.addWidget(self.distribution_start_row_label, 4, 0)
        self.main_layout.addWidget(self.distribution_start_row_input, 4, 1)
        self.distribution_start_column_label = QtWidgets.QLabel(self.tr("Distribution Start Column"))
        self.distribution_start_column_input = QtWidgets.QSpinBox()
        self.distribution_start_column_input.setRange(2, 999999)
        self.main_layout.addWidget(self.distribution_start_column_label, 5, 0)
        self.main_layout.addWidget(self.distribution_start_column_input, 5, 1)

        self.try_load_button = QtWidgets.QPushButton(self.tr("Try Load"))
        self.try_load_button.clicked.connect(self.on_try_load_clicked)
        self.try_load_button.setEnabled(False)
        self.main_layout.addWidget(self.try_load_button, 6, 0, 1, 2)

    @property
    def sheet_index(self) -> int:
        sheet_index = self.sheet_combo_box.currentIndex()
        return sheet_index
    @property
    def sheet_name(self) -> str:
        sheet_name = self.sheet_combo_box.currentText()
        return sheet_name

    @property
    def data_layout(self):
        classes_row = self.classes_row_input.value() - 1
        sample_names_column = self.sample_names_column_input.value() - 1
        distribution_start_row = self.distribution_start_row_input.value() - 1
        distribution_start_column = self.distribution_start_column_input.value() - 1
        try:
            layout = DataLayoutSetting(
                classes_row=classes_row,
                sample_names_column=sample_names_column,
                distribution_start_row=distribution_start_row,
                distribution_start_column=distribution_start_column)
            return layout
        except DataLayoutError as e:
            self.logger.exception("The current layout setting is invalid.", stack_info=True)
            self.show_error(self.tr("The current layout setting is invalid."))
            return None

    def show_message(self, title: str, message: str):
        self.normal_msg.setWindowTitle(title)
        self.normal_msg.setText(message)
        self.normal_msg.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    def on_select_clicked(self):
        filename, _ = self.file_dialog.getOpenFileName(
            self, self.tr("Select a file"), None,
            "Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)")
        if filename is None or filename == "":
            return
        self.filename = filename
        file_type = get_type_by_name(filename)
        if file_type == ReadFileType.CSV:
            sheet_names = [os.path.basename(filename)]
        elif file_type == ReadFileType.XLS:
            self.workbook = xlrd.open_workbook(filename)
            sheet_names = self.workbook.sheet_names()
        elif file_type == ReadFileType.XLSX:
            self.workbook = openpyxl.load_workbook(filename, read_only=True, data_only=True)
            sheet_names = self.workbook.sheetnames
        else:
            raise NotImplementedError(file_type)

        self.filename_display.setText(os.path.basename(filename))
        self.sheet_combo_box.clear()
        self.sheet_combo_box.addItems(sheet_names)
        self.try_load_button.setEnabled(True)

    def on_try_load_clicked(self):
        try:
            progress_dialog = QtWidgets.QProgressDialog(
                self.tr("Loading grain size distributions..."), self.tr("Cancel"),
                0, 100, self)
            progress_dialog.setWindowTitle("QGrain")
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            def callback(progress: float):
                if progress_dialog.wasCanceled():
                    raise StopIteration()
                progress_dialog.setValue(int(progress*100))
                QtCore.QCoreApplication.processEvents()
            result = load_dataset(self.filename, self.sheet_index, self.data_layout, progress_callback=callback)
            progress_dialog.setValue(100)
            if result is None:
                self.logger.exception("Can not load the grain size dataset from this file. Please check the logs for more details.", stack_info=True)
                self.show_error(self.tr("Can not load the grain size dataset from this file. Please check the logs for more details."))
            else:
                self.dataset_loaded.emit(result)
                self.logger.info("Good job! The grain size dataset has been loaded from this file, and has been emitted to other widgets.")
                self.hide()
        except StopIteration as e:
            self.logger.info("Loading task was canceled.")
            progress_dialog.close()
        except Exception as e:
            progress_dialog.close()
            self.logger.exception("An unknown exception was raised. Please check the logs for more details.", stack_info=True)
            self.show_error(self.tr("An unknown exception was raised. Please check the logs for more details."))

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("Dataset Loader"))
        if self.filename is None:
            self.filename_display.setText(self.tr("Filename Unspecified"))
            self.sheet_combo_box.setItemText(0, self.tr("Empty"))
        self.select_button.setText(self.tr("Select"))
        self.sheet_label.setText(self.tr("Sheet Name"))
        self.classes_row_label.setText(self.tr("Row With Grain Size Classes"))
        self.sample_names_column_label.setText(self.tr("Column With Sample Names"))
        self.distribution_start_row_label.setText(self.tr("Distribution Start Row"))
        self.distribution_start_column_label.setText(self.tr("Distribution Start Column"))
        self.try_load_button.setText(self.tr("Try Load"))
