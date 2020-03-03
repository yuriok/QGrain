__all__ = ["DataSetting"]

import logging
import os

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import (QCheckBox, QGridLayout, QLabel, QMessageBox,
                               QSpinBox, QWidget)

from QGrain.models.DataLayoutSetting import DataLayoutError, DataLayoutSetting

QGRAIN_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

class DataSetting(QWidget):
    sigDataSettingChanged = Signal(dict)
    logger = logging.getLogger("root.ui.DataSetting")
    gui_logger = logging.getLogger("GUI")
    def __init__(self):
        super().__init__()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.settings = QSettings(os.path.join(QGRAIN_ROOT_PATH, "settings", "QGrain.ini"), QSettings.Format.IniFormat)
        self.settings.beginGroup("data")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)

        self.title_label = QLabel(self.tr("Data Settings:"))
        self.title_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.title_label, 0, 0)

        self.class_row_label = QLabel(self.tr("Class Row"))
        self.class_row_label.setToolTip(self.tr("The row index (starts with 0) of grain size classes."))
        self.main_layout.addWidget(self.class_row_label, 1, 0)
        self.class_row_input = QSpinBox()
        self.class_row_input.setMinimum(0)
        self.main_layout.addWidget(self.class_row_input, 1, 1)

        self.sample_name_col_label = QLabel(self.tr("Sample Name Column"))
        self.sample_name_col_label.setToolTip(self.tr("The column index (starts with 0) of sample names."))
        self.main_layout.addWidget(self.sample_name_col_label, 2, 0)
        self.sample_name_col_input = QSpinBox()
        self.sample_name_col_input.setMinimum(0)
        self.main_layout.addWidget(self.sample_name_col_input, 2, 1)

        self.distribution_start_row_label = QLabel(self.tr("Distribution Start Row"))
        self.distribution_start_row_label.setToolTip(self.tr("The start row index (starts with 0) of distribution data.\nIt should be greater than the row index of classes."))
        self.main_layout.addWidget(self.distribution_start_row_label, 3, 0)
        self.distribution_start_row_input = QSpinBox()
        self.distribution_start_row_input.setMinimum(1)
        self.main_layout.addWidget(self.distribution_start_row_input, 3, 1)

        self.distribution_start_col_label = QLabel(self.tr("Distribution Start Column"))
        self.distribution_start_col_label.setToolTip(self.tr("The start column index (starts with 0) of distribution data.\nIt should be greater than the column index of sample name."))
        self.main_layout.addWidget(self.distribution_start_col_label, 4, 0)
        self.distribution_start_col_input = QSpinBox()
        self.distribution_start_col_input.setMinimum(1)
        self.main_layout.addWidget(self.distribution_start_col_input, 4, 1)

        self.draw_charts_checkbox = QCheckBox(self.tr("Draw Charts"))
        self.draw_charts_checkbox.setChecked(True)
        self.draw_charts_checkbox.setToolTip(self.tr("Whether to draw charts while saving .xlsx file.\nIf the samples are too many, the massive charts will slow the running of Excel heavily."))
        self.main_layout.addWidget(self.draw_charts_checkbox, 5, 0, 1, 2)

        self.restore()

        self.class_row_input.valueChanged.connect(self.on_loading_setting_changed)
        self.sample_name_col_input.valueChanged.connect(self.on_loading_setting_changed)
        self.distribution_start_row_input.valueChanged.connect(self.on_loading_setting_changed)
        self.distribution_start_col_input.valueChanged.connect(self.on_loading_setting_changed)

    def on_loading_setting_changed(self):
        # read settings from ui
        classes_row = self.class_row_input.value()
        sample_name_column = self.sample_name_col_input.value()
        distribution_start_row = self.distribution_start_row_input.value()
        distribution_start_column = self.distribution_start_col_input.value()
        # use the ctor of `DataLayoutSetting` to check its validation, and then emit signal
        try:
            layout = DataLayoutSetting(classes_row,
                                       sample_name_column,
                                       distribution_start_row,
                                       distribution_start_column)
            self.sigDataSettingChanged.emit({"layout": layout})
        except DataLayoutError:
            self.logger.exception("The data layout setting is invalid.", stack_info=True)
            self.msg_box.setWindowTitle(self.tr("Error"))
            self.msg_box.setText(self.tr("Invalid data layout setting."))
            self.msg_box.exec_()
            return
        self.settings.setValue("classes_row", classes_row)
        self.settings.setValue("sample_name_column", sample_name_column)
        self.settings.setValue("distribution_start_row", distribution_start_row)
        self.settings.setValue("distribution_start_column", distribution_start_column)

    def on_draw_charts_changed(self):
        if self.draw_charts_checkbox.checkState() == Qt.Checked:
            self.settings.setValue("draw_charts", True)
            self.sigDataSettingChanged.emit(dict(draw_charts=True))
        else:
            self.settings.setValue("draw_charts", True)
            self.sigDataSettingChanged.emit(dict(draw_charts=False))

    def restore(self):
        self.class_row_input.setValue(self.settings.value("classes_row", defaultValue=0, type=int))
        self.sample_name_col_input.setValue(self.settings.value("sample_name_column", defaultValue=0, type=int))
        self.distribution_start_row_input.setValue(self.settings.value("distribution_start_row", defaultValue=1, type=int))
        self.distribution_start_col_input.setValue(self.settings.value("distribution_start_column", defaultValue=1, type=int))

        if self.settings.value("draw_charts", defaultValue=True, type=bool):
            self.draw_charts_checkbox.setCheckState(Qt.Checked)
        else:
            self.draw_charts_checkbox.setCheckState(Qt.Unchecked)


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main = DataSetting()
    main.show()
    sys.exit(app.exec_())
