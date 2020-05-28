__all__ = ["DataSettingWidget"]

import logging

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import (QCheckBox, QGridLayout, QLabel, QMessageBox,
                               QSpinBox, QWidget)

from QGrain.models.DataLayoutSettings import (DataLayoutError,
                                              DataLayoutSettings)


class DataSettingWidget(QWidget):
    data_settings_changed_signal = Signal(dict)
    logger = logging.getLogger("root.ui.DataSettingWidget")
    gui_logger = logging.getLogger("GUI")
    def __init__(self, filename: str = None, group: str = None):
        super().__init__()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        if filename is not None:
            self.setting_file = QSettings(filename, QSettings.Format.IniFormat)
            if group is not None:
                self.setting_file.beginGroup(group)
        else:
            self.setting_file = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.initialize_ui()

    def initialize_ui(self):
        self.main_layout = QGridLayout(self)

        self.class_row_label = QLabel(self.tr("Class Row"))
        self.class_row_label.setToolTip(self.tr("The row index (starts with 0) of grain size classes."))
        self.main_layout.addWidget(self.class_row_label, 0, 0)
        self.class_row_input = QSpinBox()
        self.class_row_input.setMinimum(0)
        self.class_row_input.setValue(0)
        self.main_layout.addWidget(self.class_row_input, 0, 1)

        self.sample_name_col_label = QLabel(self.tr("Sample Name Column"))
        self.sample_name_col_label.setToolTip(self.tr("The column index (starts with 0) of sample names."))
        self.main_layout.addWidget(self.sample_name_col_label, 1, 0)
        self.sample_name_col_input = QSpinBox()
        self.sample_name_col_input.setMinimum(0)
        self.sample_name_col_input.setValue(0)
        self.main_layout.addWidget(self.sample_name_col_input, 1, 1)

        self.distribution_start_row_label = QLabel(self.tr("Distribution Start Row"))
        self.distribution_start_row_label.setToolTip(self.tr("The start row index (starts with 0) of distribution data.\nIt should be greater than the row index of classes."))
        self.main_layout.addWidget(self.distribution_start_row_label, 2, 0)
        self.distribution_start_row_input = QSpinBox()
        self.distribution_start_row_input.setMinimum(1)
        self.distribution_start_row_input.setValue(1)
        self.main_layout.addWidget(self.distribution_start_row_input, 2, 1)

        self.distribution_start_col_label = QLabel(self.tr("Distribution Start Column"))
        self.distribution_start_col_label.setToolTip(self.tr("The start column index (starts with 0) of distribution data.\nIt should be greater than the column index of sample name."))
        self.main_layout.addWidget(self.distribution_start_col_label, 3, 0)
        self.distribution_start_col_input = QSpinBox()
        self.distribution_start_col_input.setMinimum(1)
        self.distribution_start_col_input.setValue(1)
        self.main_layout.addWidget(self.distribution_start_col_input, 3, 1)

        self.draw_charts_checkbox = QCheckBox(self.tr("Draw Charts"))
        self.draw_charts_checkbox.setChecked(True)
        self.draw_charts_checkbox.setToolTip(self.tr("Whether to draw charts while saving .xlsx file.\nIf the samples are too many, the massive charts will slow the running of Excel heavily."))
        self.main_layout.addWidget(self.draw_charts_checkbox, 4, 0, 1, 2)

        self.class_row_input.valueChanged.connect(self.on_loading_setting_changed)
        self.sample_name_col_input.valueChanged.connect(self.on_loading_setting_changed)
        self.distribution_start_row_input.valueChanged.connect(self.on_loading_setting_changed)
        self.distribution_start_col_input.valueChanged.connect(self.on_loading_setting_changed)


    @property
    def data_layout_settings(self):
        # read settings from ui
        classes_row = self.class_row_input.value()
        sample_name_column = self.sample_name_col_input.value()
        distribution_start_row = self.distribution_start_row_input.value()
        distribution_start_column = self.distribution_start_col_input.value()
        # use the ctor of `DataLayoutSettings` to check its validation, and then emit signal
        try:
            layout = DataLayoutSettings(classes_row,
                                       sample_name_column,
                                       distribution_start_row,
                                       distribution_start_column)
            return layout
        except DataLayoutError:
            self.logger.exception("The data layout setting is invalid.", stack_info=True)
            return None

    def save(self):
        if self.setting_file is not None:
            if self.draw_charts_checkbox.checkState() == Qt.Checked:
                self.setting_file.setValue("draw_charts", True)
            else:
                self.setting_file.setValue("draw_charts", True)
            
            layout = self.data_layout_settings
            if layout is None:
                return
            else:
                self.setting_file.setValue("classes_row", layout.classes_row)
                self.setting_file.setValue("sample_name_column", layout.sample_name_column)
                self.setting_file.setValue("distribution_start_row", layout.distribution_start_row)
                self.setting_file.setValue("distribution_start_column", layout.distribution_start_column)

            self.logger.info("Data settings have been saved to the file.")

    def restore(self):
        if self.setting_file is not None:
            self.class_row_input.setValue(self.setting_file.value("classes_row", defaultValue=0, type=int))
            self.sample_name_col_input.setValue(self.setting_file.value("sample_name_column", defaultValue=0, type=int))
            self.distribution_start_row_input.setValue(self.setting_file.value("distribution_start_row", defaultValue=1, type=int))
            self.distribution_start_col_input.setValue(self.setting_file.value("distribution_start_column", defaultValue=1, type=int))

            if self.setting_file.value("draw_charts", defaultValue=True, type=bool):
                self.draw_charts_checkbox.setCheckState(Qt.Checked)
            else:
                self.draw_charts_checkbox.setCheckState(Qt.Unchecked)

            self.logger.info("Data settings have been retored from the file.")

    def on_loading_setting_changed(self):
        layout = self.data_layout_settings
        if layout is None:
            self.msg_box.setWindowTitle(self.tr("Error"))
            self.msg_box.setText(self.tr("Invalid data layout setting."))
            self.msg_box.exec_()
        else:
            self.data_settings_changed_signal.emit({"layout": layout})

        self.logger.info("Data layout settings have been changed.")

    def on_draw_charts_changed(self):
        if self.draw_charts_checkbox.checkState() == Qt.Checked:
            self.data_settings_changed_signal.emit(dict(draw_charts=True))
        else:
            self.data_settings_changed_signal.emit(dict(draw_charts=False))

        self.logger.info("Data setting [Draw Charts] has been changed.")


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main = DataSettingWidget()
    main.show()
    sys.exit(app.exec_())
