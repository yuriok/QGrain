__all__ = ["DatasetLoader"]

import logging
import os
import typing

import openpyxl
import xlrd
from PySide6 import QtCore, QtWidgets

from ..models import Dataset
from ..io.load_dataset import check_layout, load_dataset, get_file_type, ReadFileType


class DatasetLoader(QtWidgets.QDialog):
    logger = logging.getLogger("QGrain.DatasetLoader")
    dataset_loaded = QtCore.Signal(Dataset)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setWindowTitle(self.tr("Dataset Loader"))
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
        self.class_row_label = QtWidgets.QLabel(self.tr("Row With Grain Size Classes"))
        self.class_row_input = QtWidgets.QSpinBox()
        self.class_row_input.setRange(1, 999)
        self.main_layout.addWidget(self.class_row_label, 2, 0)
        self.main_layout.addWidget(self.class_row_input, 2, 1)
        self.name_column_label = QtWidgets.QLabel(self.tr("Column With Sample Names"))
        self.name_column_input = QtWidgets.QSpinBox()
        self.name_column_input.setRange(1, 999)
        self.main_layout.addWidget(self.name_column_label, 3, 0)
        self.main_layout.addWidget(self.name_column_input, 3, 1)
        self.start_row_label = QtWidgets.QLabel(self.tr("Start Row of Distributions"))
        self.start_row_input = QtWidgets.QSpinBox()
        self.start_row_input.setRange(2, 999999)
        self.main_layout.addWidget(self.start_row_label, 4, 0)
        self.main_layout.addWidget(self.start_row_input, 4, 1)
        self.start_column_label = QtWidgets.QLabel(self.tr("Start Column of Distributions"))
        self.start_column_input = QtWidgets.QSpinBox()
        self.start_column_input.setRange(2, 999999)
        self.main_layout.addWidget(self.start_column_label, 5, 0)
        self.main_layout.addWidget(self.start_column_input, 5, 1)
        self.try_load_button = QtWidgets.QPushButton(self.tr("Try Load"))
        self.try_load_button.clicked.connect(self.on_try_load_clicked)
        self.try_load_button.setEnabled(False)
        self.main_layout.addWidget(self.try_load_button, 6, 0, 1, 2)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self._filename = ""

    @property
    def sheet_index(self) -> int:
        sheet_index = self.sheet_combo_box.currentIndex()
        return sheet_index

    @property
    def sheet_name(self) -> str:
        sheet_name = self.sheet_combo_box.currentText()
        return sheet_name

    @property
    def data_layout(self) -> typing.Optional[typing.Dict[str, int]]:
        class_row = self.class_row_input.value() - 1
        name_col = self.name_column_input.value() - 1
        start_row = self.start_row_input.value() - 1
        start_col = self.start_column_input.value() - 1
        layout = dict(class_row=class_row, name_col=name_col, start_row=start_row, start_col=start_col)
        return layout

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
            self, self.tr("Select a file"), ".",
            "Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)")
        if filename is None or filename == "":
            return
        self._filename = filename
        file_type = get_file_type(filename)
        if file_type == ReadFileType.CSV:
            sheet_names = [os.path.basename(filename)]
        elif file_type == ReadFileType.XLS:
            workbook = xlrd.open_workbook(filename)
            sheet_names = workbook.sheet_names()
            workbook.release_resources()
        elif file_type == ReadFileType.XLSX:
            workbook = openpyxl.load_workbook(filename, read_only=True, data_only=True)
            sheet_names = workbook.sheetnames
            workbook.close()
        else:
            raise NotImplementedError(file_type)
        self.filename_display.setText(os.path.basename(filename))
        self.sheet_combo_box.clear()
        self.sheet_combo_box.addItems(sheet_names)
        self.try_load_button.setEnabled(True)

    def on_try_load_clicked(self):
        try:
            check_layout(**self.data_layout)
        except ValueError:
            self.logger.error("The current layout setting is invalid.")
            self.show_error(self.tr("The current layout setting is invalid."))
            return

        progress_dialog = QtWidgets.QProgressDialog(
            self.tr("Loading the grain size distributions..."), self.tr("Cancel"),
            0, 100, self)
        progress_dialog.setWindowTitle("QGrain")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)

        def callback(progress: float):
            if progress_dialog.wasCanceled():
                raise StopIteration()
            progress_dialog.setValue(int(progress * 100))
            QtCore.QCoreApplication.processEvents()
        try:
            result = load_dataset(self._filename, sheet_index=self.sheet_index,  **self.data_layout,
                                  progress_callback=callback, logger=self.logger)
            if result is None:
                self.show_error(self.tr("Can not load the grain size dataset from this file. "
                                        "Please check the logs for more details."))
            else:
                progress_dialog.setValue(100)
                self.dataset_loaded.emit(result)
                self.logger.info("Good job! The grain size dataset has been loaded from this file, "
                                 "and has been emitted to other widgets.")
                self.hide()
        except StopIteration:
            self.logger.info("Loading task was canceled.")
        finally:
            progress_dialog.close()

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("Dataset Loader"))
        if self._filename is None:
            self.filename_display.setText(self.tr("Filename Unspecified"))
            self.sheet_combo_box.setItemText(0, self.tr("Empty"))
        self.select_button.setText(self.tr("Select"))
        self.sheet_label.setText(self.tr("Sheet Name"))
        self.class_row_label.setText(self.tr("Row With Grain Size Classes"))
        self.name_column_label.setText(self.tr("Column With Sample Names"))
        self.start_row_label.setText(self.tr("Start Row of Distributions"))
        self.start_column_label.setText(self.tr("Start Column of Distributions"))
        self.try_load_button.setText(self.tr("Try Load"))
