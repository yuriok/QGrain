from PySide2.QtWidgets import QMainWindow, QCheckBox, QLabel, QRadioButton, QPushButton, QGridLayout, QApplication, QSizePolicy, QWidget, QTabWidget, QComboBox, QLineEdit, QMessageBox
from PySide2.QtCore import Qt, QSettings, Signal
from PySide2.QtGui import QIcon, QValidator, QIntValidator

import logging

class DataSetting(QWidget):
    sigDataLoaderSettingChanged = Signal(dict)
    sigDataWriterSettingChanged = Signal(dict)
    logger = logging.getLogger("root.ui.DataSetting")
    gui_logger = logging.getLogger("GUI")
    def __init__(self, settings: QSettings):
        super().__init__()
        self.settings = settings
        self.msg_box = QMessageBox()
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.int_validator = QIntValidator()
        self.int_validator.setBottom(0)

        self.class_row_label = QLabel(self.tr("Class Row"))
        self.class_row_label.setToolTip(self.tr("The row index (starts with 0) of grain size classes that stored in data file."))
        self.main_layout.addWidget(self.class_row_label, 0, 0)
        self.class_row_edit = QLineEdit()
        self.class_row_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.class_row_edit, 0, 1)

        self.sample_name_col_label = QLabel(self.tr("Sample Name Column"))
        self.sample_name_col_label.setToolTip(self.tr("The column index (starts with 0) of sample names that stored in data file."))
        self.main_layout.addWidget(self.sample_name_col_label, 1, 0)
        self.sample_name_col_edit = QLineEdit()
        self.sample_name_col_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.sample_name_col_edit, 1, 1)

        self.data_start_row_label = QLabel(self.tr("Data Start Row"))
        self.data_start_row_label.setToolTip(self.tr("The start row index (starts with 0) of sample data that stored in data file.\nIt should be greater than the index of Class Row."))
        self.main_layout.addWidget(self.data_start_row_label, 2, 0)
        self.data_start_row_edit = QLineEdit()
        self.data_start_row_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.data_start_row_edit, 2, 1)

        self.data_start_col_label = QLabel(self.tr("Data Start Column"))
        self.data_start_col_label.setToolTip(self.tr("The start column index (starts with 0) of sample data that stored in data file.\nIt should be greater than the index of Sample Name Column."))
        self.main_layout.addWidget(self.data_start_col_label, 3, 0)
        self.data_start_col_edit = QLineEdit()
        self.data_start_col_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.data_start_col_edit, 3, 1)

        self.draw_charts_checkbox = QCheckBox(self.tr("Draw Charts"))
        self.draw_charts_checkbox.setChecked(True)
        self.draw_charts_checkbox.setToolTip(self.tr("This option controls whether it will draw charts when saving data as xlsx file.\nIf your samples are too many, the massive charts will slow the excel runing heavily.\nThen you can disable this option or save your data separately."))
        self.main_layout.addWidget(self.draw_charts_checkbox, 4, 0, 1, 2)

    def save_settings(self, settings:QSettings):
        settings.beginGroup("data_loading")
        # read settings from ui
        classes_row = self.class_row_edit.text()
        sample_name_column = self.sample_name_col_edit.text()
        data_start_row = self.data_start_row_edit.text()
        data_start_column = self.data_start_col_edit.text()
        # set values to `QSetting`
        # note, the values set to `QSettings` must be str to avoid exceptions
        # because if use int or other type of values, the types of values from current `QSettings` and the `ini` file are not equal
        # `ini` files only yield `str`, and current `QSettings` will store the original type of values
        settings.setValue("classes_row", classes_row)
        settings.setValue("sample_name_column", sample_name_column)
        settings.setValue("data_start_row", data_start_row)
        settings.setValue("data_start_column", data_start_column)
        # emit signal
        try:
            signal_data = dict(data_layout=dict(classes_row=int(classes_row), sample_name_column=int(sample_name_column), data_start_row=int(data_start_row), data_start_column=int(data_start_column)))
            self.sigDataLoaderSettingChanged.emit(signal_data)
        # raise while converting invalid `str` to `int`
        except ValueError:
            self.logger.exception("Some unknown exception raised, maybe the `QLineEdit` widget has not a valid `QValidator`.", stack_info=True)
            self.gui_logger.error(self.tr("Some unknown exception raised. Settings of data loading did not be saved."))
            # this exception raise when the `str` values can not be converted to int
            # that means the `ini` file maybe modified incorrectly
            self.msg_box.setWindowTitle(self.tr("Error"))
            self.msg_box.setText(self.tr("Some unknown exception raised. Settings of data loading did not be saved."))
            self.msg_box.exec_()
        finally:
            settings.endGroup()

        settings.beginGroup("data_saving")
        if self.draw_charts_checkbox.checkState() == Qt.Checked:
            settings.setValue("draw_charts", "True")
            self.sigDataWriterSettingChanged.emit(dict(draw_charts=True))
        else:
            settings.setValue("draw_charts", "False")
            self.sigDataWriterSettingChanged.emit(dict(draw_charts=False))
        
        settings.endGroup()

    def restore_settings(self, settings:QSettings):
        settings.beginGroup("data_loading")
        try:
            self.class_row_edit.setText(settings.value("classes_row"))
            self.sample_name_col_edit.setText(settings.value("sample_name_column"))
            self.data_start_row_edit.setText(settings.value("data_start_row"))
            self.data_start_col_edit.setText(settings.value("data_start_column"))
        except Exception:
            self.logger.exception("Unknown exception occurred. Maybe the type of values which were set to `QSettings` is not `str`.", stack_info=True)
            self.gui_logger.error(self.tr("Some unknown exception raised. Settings of data loading did not be restored."))
            # this exception raise when the values are not `str`
            # that means there are some bugs in the set progress
            self.msg_box.setWindowTitle(self.tr("Error"))
            self.msg_box.setText(self.tr("Some unknown exception raised. Settings of data loading did not be restored."))
            self.msg_box.exec_()
        finally:
            settings.endGroup()

        settings.beginGroup("data_saving")
        if settings.value("draw_charts")=="True":
            self.draw_charts_checkbox.setCheckState(Qt.Checked)
        else:
            self.draw_charts_checkbox.setCheckState(Qt.Unchecked)
        settings.endGroup()
