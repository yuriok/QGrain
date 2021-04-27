import csv
import datetime
import os
import typing
from enum import Enum, unique

import numpy as np
import openpyxl
import qtawesome as qta
import xlrd
from PySide2.QtCore import QCoreApplication, Qt, Signal
from PySide2.QtWidgets import (QComboBox, QDialog, QFileDialog, QGridLayout,
                               QLabel, QPushButton, QSpinBox, QTextEdit)
from QGrain.models.DataLayoutSetting import DataLayoutError, DataLayoutSetting
from QGrain.models.GrainSizeDataset import GrainSizeDataset


@unique
class ReadFileType(Enum):
    XLS = 0
    XLSX = 1
    CSV = 2


def get_type_by_name(filename: str):
    _, extension = os.path.splitext(filename)
    if extension == ".csv":
        return ReadFileType.CSV
    elif extension == ".xls":
        return ReadFileType.XLS
    elif extension == ".xlsx":
        return ReadFileType.XLSX
    else:
        raise NotImplementedError(extension)

class LoadDatasetDialog(QDialog):
    dataset_loaded = Signal(GrainSizeDataset)
    def __init__(self, parent=None):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Dataset Loader"))
        self.initialize_ui()
        self.file_dialog = QFileDialog(parent=self)
        self.filename = None # type: str
        self.workbook = None # type: typing.Union[openpyxl.Workbook, xlrd.Book]
        self.dataset = None # type: GrainSizeDataset

    def initialize_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.main_layout = QGridLayout(self)

        self.filename_label = QLabel(self.tr("Filename:"))
        self.filename_display = QLabel(self.tr("Unknown"))
        self.main_layout.addWidget(self.filename_label, 0, 0)
        self.main_layout.addWidget(self.filename_display, 0, 1, 1, 2)
        self.select_button = QPushButton(qta.icon("mdi.file-table"), self.tr("Select"))
        self.select_button.clicked.connect(self.on_select_clicked)
        self.main_layout.addWidget(self.select_button, 0, 3)
        self.sheet_label = QLabel(self.tr("Sheet:"))
        self.sheet_combo_box = QComboBox()
        self.sheet_combo_box.addItem(self.tr("Empty"))
        self.main_layout.addWidget(self.sheet_label, 1, 0)
        self.main_layout.addWidget(self.sheet_combo_box, 1, 1, 1, 3)

        self.classes_row_label = QLabel(self.tr("Row With Grain-size Classes:"))
        self.classes_row_input = QSpinBox()
        self.classes_row_input.setRange(1, 999)
        self.main_layout.addWidget(self.classes_row_label, 2, 0, 1, 3)
        self.main_layout.addWidget(self.classes_row_input, 2, 3)
        self.sample_names_column_label = QLabel(self.tr("Column With Sample Names:"))
        self.sample_names_column_input = QSpinBox()
        self.sample_names_column_input.setRange(1, 999)
        self.main_layout.addWidget(self.sample_names_column_label, 3, 0, 1, 3)
        self.main_layout.addWidget(self.sample_names_column_input, 3, 3)
        self.distribution_start_row_label = QLabel(self.tr("Distribution Start Row:"))
        self.distribution_start_row_input = QSpinBox()
        self.distribution_start_row_input.setRange(2, 999999)
        self.main_layout.addWidget(self.distribution_start_row_label, 4, 0, 1, 3)
        self.main_layout.addWidget(self.distribution_start_row_input, 4, 3)
        self.distribution_start_column_label = QLabel(self.tr("Distribution Start Column:"))
        self.distribution_start_column_input = QSpinBox()
        self.distribution_start_column_input.setRange(2, 999999)
        self.main_layout.addWidget(self.distribution_start_column_label, 5, 0, 1, 3)
        self.main_layout.addWidget(self.distribution_start_column_input, 5, 3)

        self.try_load_button = QPushButton(qta.icon("fa5s.book-reader"), self.tr("Try Load"))
        self.try_load_button.clicked.connect(self.on_try_load_clicked)
        self.try_load_button.setEnabled(False)
        self.main_layout.addWidget(self.try_load_button, 6, 0, 1, 4)

        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.main_layout.addWidget(self.info_display, 7, 0, 1, 4)

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
            layout = DataLayoutSetting(classes_row=classes_row,
                                    sample_names_column=sample_names_column,
                                    distribution_start_row=distribution_start_row,
                                    distribution_start_column=distribution_start_column)
            return layout
        except DataLayoutError as e:
            self.show_error(f"The current setting is invalid.\n    {e.__str__()}")
            return None

    def on_select_clicked(self):
        filename, _ = self.file_dialog.getOpenFileName(\
            self, self.tr("Select a file"), None,
            self.tr("Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)"))
        if filename is None or filename == "":
            self.show_warning(f"No file was selected.")
            return
        self.filename = filename
        file_type = get_type_by_name(filename)
        self.show_info(f"Data file [{file_type}] was selected: [{filename}].")
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

        if file_type != ReadFileType.CSV:
            self.show_info(f"It has {len(sheet_names)} sheet(s), please select one.")
        self.filename_display.setText(os.path.basename(filename))
        self.sheet_combo_box.clear()
        self.sheet_combo_box.addItems(sheet_names)
        self.try_load_button.setEnabled(True)

    def show_info(self, text: str):
        self.info_display.append(f'<font size="3" color="black">[{datetime.datetime.now()}] - {text}</font>\n')

    def show_warning(self, text: str):
        self.info_display.append(f'<font size="3" color="#fed71a">[{datetime.datetime.now()}] - {text}</font>\n')

    def show_error(self, text: str):
        self.info_display.append(f'<font size="3" color="#f03752">[{datetime.datetime.now()}] - {text}</font>\n')

    def show_success(self, text: str):
        self.info_display.append(f'<font size="3" color="#2c9678">[{datetime.datetime.now()}] - {text}</font>\n')

    def on_try_load_clicked(self):
        try:
            self.try_load()
        except Exception as e:
            self.show_error(f"Error raised while loading.\n    {e.__str__()}")

    def try_load(self):
        layout = self.data_layout
        if layout is None:
            return
        assert self.filename is not None
        file_type = get_type_by_name(self.filename)
        self.show_info("Start to load raw data from the file.")
        QCoreApplication.processEvents()
        if file_type == ReadFileType.CSV:
            try:
                with open(self.filename, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    raw_data = [row for row in reader]
            except Exception as e:
                self.show_error(f"Exception rasised when reading {file_type} file.\n    {e.__str__()}")
                return
        elif file_type == ReadFileType.XLS:
            sheet = self.workbook.sheet_by_index(self.sheet_index)
            raw_data = [sheet.row_values(row) for row in range(sheet.nrows)]
        elif file_type == ReadFileType.XLSX:
            sheet = self.workbook[self.sheet_name]
            raw_data = [[value for value in row] for row in sheet.values]
        else:
            raise NotImplementedError(file_type)
        self.show_success(f"Raw data has been loaded from the file.")
        QCoreApplication.processEvents()
        try:
            classes_μm = np.array(raw_data[layout.classes_row][layout.distribution_start_column:], dtype=np.float64)
        except Exception as e:
            self.show_error(f"Can not convert the row of classes to a numerical array, it may contains invalid values (e.g. text or empty cell).\n    {e.__str__()}")
            return
        self.show_info(f"Grain size classes in μm: [{','.join([f'{x: 0.4f}' for x in classes_μm[:3]])}, ...,{','.join([f'{x: 0.4f}' for x in classes_μm[-3:]])}].")
        GrainSizeDataset.validate_classes_μm(classes_μm)
        self.show_success("Validation of grain size classes passed.")
        QCoreApplication.processEvents()

        names = []
        distributions = []
        i = layout.distribution_start_row + 1
        for row_values in raw_data[layout.distribution_start_row:]:
            # check if it's a empty row, i.e. the values all are empty string
            is_empty_row = True
            for distribution_value in row_values[layout.distribution_start_column:]:
                if distribution_value != "" and distribution_value is not None:
                    is_empty_row = False
                    break
            # if it's a empty row, jump this row to process the next one
            if is_empty_row:
                self.show_warning(f"This row is empty, jump to next.")
                continue

            sample_name = row_values[layout.sample_name_column]
            self.show_info(f"Processing the {i} row, sample name is [{sample_name}].")
            if sample_name is None:
                sample_name = "NONE"
                self.show_warning(f"The sample name is invalid, use 'NONE' instead.")
            # users may use pure number as the sample name
            elif type(sample_name) != str:
                sample_name = str(sample_name)
                self.show_warning(f"The sample name is not text (may be a number), convert it to text.")
            elif sample_name == "":
                sample_name = "EMPTY"
                self.show_warning(f"The sample name is a empty text, use 'EMPTY' instead.")

            try:
                distribution = np.array(row_values[layout.distribution_start_column:], dtype=np.float64)
            except Exception as e:
                self.show_error(f"Can not convert the distribution values at row [{i}] to a numerical array, it may contains invalid values (e.g. text or empty cell).\n    {e.__str__()}")
                return
            try:
                GrainSizeDataset.validate_distribution(distribution)
            except Exception as e:
                self.show_error(f"Validation of the distribution array of sample [{sample_name}] did not pass.\n    {e.__str__()}")
                return
            names.append(sample_name)
            distributions.append(distribution)
            i += 1
            self.show_info(f"Validation of sample {sample_name} passed.")
            QCoreApplication.processEvents()

        self.show_success("All data has been convert to array. Next to do the final validation.")
        dataset = GrainSizeDataset()
        dataset.add_batch(classes_μm, names, distributions)
        self.dataset = dataset
        self.show_success("Dataset has been loaded successfully, you can close this dialog now.")
        self.dataset_loaded.emit(dataset)


if __name__ == "__main__":
    import sys
    from QGrain.entry import setup_app
    app = setup_app()
    main = LoadDatasetDialog()
    main.show()
    sys.exit(app.exec_())
