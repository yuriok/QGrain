__all__ = ["AlgorithmSettingWidget"]

import logging
import os

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtWidgets import (QDoubleSpinBox, QFrame, QGridLayout, QLabel,
                               QMessageBox, QSpinBox, QWidget)

from QGrain.models.AlgorithmSettings import AlgorithmSettings


class AlgorithmSettingWidget(QWidget):
    setting_changed_signal = Signal(AlgorithmSettings)
    logger = logging.getLogger("root.ui.AlgorithmSettingWidget")
    gui_logger = logging.getLogger("GUI")
    def __init__(self, filename: str = None, group: str = None):
        super().__init__()
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
        self.global_optimization_maximum_iteration_label = QLabel(self.tr("Global Optimization Maximum Iteration"))
        self.global_optimization_maximum_iteration_label.setToolTip(self.tr("Maximum iteration number of global optimization.\nIf the iteration number of global optimization has reached the maximum, fitting process will terminate."))
        self.main_layout.addWidget(self.global_optimization_maximum_iteration_label, 0, 0)
        self.global_optimization_maximum_iteration_input = QSpinBox()
        self.global_optimization_maximum_iteration_input.setRange(0, 10000)
        self.global_optimization_maximum_iteration_input.setValue(100)
        self.main_layout.addWidget(self.global_optimization_maximum_iteration_input, 0, 1)

        self.global_optimization_success_iteration_label = QLabel(self.tr("Global Optimization Success Iteration"))
        self.global_optimization_success_iteration_label.setToolTip(self.tr("The iteration number of reaching the same local optimal value.\nIt's another termination condition of global optimization."))
        self.main_layout.addWidget(self.global_optimization_success_iteration_label, 1, 0)
        self.global_optimization_success_iteration_input = QSpinBox()
        self.global_optimization_success_iteration_input.setRange(0, 100)
        self.global_optimization_success_iteration_input.setValue(3)
        self.main_layout.addWidget(self.global_optimization_success_iteration_input, 1, 1)

        self.global_optimization_step_size_label = QLabel(self.tr("Global Optimization Step Size"))
        self.global_optimization_step_size_label.setToolTip(self.tr("The step size of searching global optimal value.\nGreater step size will jump out the local optimal value easier but may miss the global optimal value."))
        self.main_layout.addWidget(self.global_optimization_step_size_label, 2, 0)
        self.global_optimization_step_size_input = QDoubleSpinBox()
        self.global_optimization_step_size_input.setRange(0.0001, 10000)
        self.global_optimization_step_size_input.setValue(1.0)
        self.main_layout.addWidget(self.global_optimization_step_size_input, 2, 1)

        self.global_optimization_minimizer_tolerance_level_label = QLabel(self.tr("Global Optimization Minimizer Tolerance Level"))
        self.global_optimization_minimizer_tolerance_level_label.setToolTip(self.tr("The tolerance level of the minimizer of global optimization.\nTolerance level means the accepted minimum variation (10 ^ -level) of the target function.\nIt controls the precision and and influences the speed."))
        self.main_layout.addWidget(self.global_optimization_minimizer_tolerance_level_label, 3, 0)
        self.global_optimization_minimizer_tolerance_level_input = QSpinBox()
        self.global_optimization_minimizer_tolerance_level_input.setRange(1, 1000)
        self.global_optimization_minimizer_tolerance_level_input.setValue(8)
        self.main_layout.addWidget(self.global_optimization_minimizer_tolerance_level_input, 3, 1)

        self.global_optimization_minimizer_maximum_iteration_label = QLabel(self.tr("Global Optimization Minimizer Maximum Iteration"))
        self.global_optimization_minimizer_maximum_iteration_label.setToolTip(self.tr("Maximum iteration number of the minimizer of global optimization."))
        self.main_layout.addWidget(self.global_optimization_minimizer_maximum_iteration_label, 4, 0)
        self.global_optimization_minimizer_maximum_iteration_input = QSpinBox()
        self.global_optimization_minimizer_maximum_iteration_input.setRange(0, 10000)
        self.global_optimization_minimizer_maximum_iteration_input.setValue(200)
        self.main_layout.addWidget(self.global_optimization_minimizer_maximum_iteration_input, 4, 1)

        self.final_optimization_minimizer_tolerance_level_label = QLabel(self.tr("Final Optimization Minimizer Tolerance Level"))
        self.final_optimization_minimizer_tolerance_level_label.setToolTip(self.tr("The tolerance level of the minimizer of final optimization."))
        self.main_layout.addWidget(self.final_optimization_minimizer_tolerance_level_label, 5, 0)
        self.final_optimization_minimizer_tolerance_level_input = QSpinBox()
        self.final_optimization_minimizer_tolerance_level_input.setRange(1, 1000)
        self.final_optimization_minimizer_tolerance_level_input.setValue(100)
        self.main_layout.addWidget(self.final_optimization_minimizer_tolerance_level_input, 5, 1)

        self.final_optimization_minimizer_maximum_iteration_label = QLabel(self.tr("Final Optimization Maximum Iteration"))
        self.final_optimization_minimizer_maximum_iteration_label.setToolTip(self.tr("Maximum iteration number of the minimizer of final optimization."))
        self.main_layout.addWidget(self.final_optimization_minimizer_maximum_iteration_label, 6, 0)
        self.final_optimization_minimizer_maximum_iteration_input = QSpinBox()
        self.final_optimization_minimizer_maximum_iteration_input.setRange(100, 10000)
        self.final_optimization_minimizer_maximum_iteration_input.setValue(1000)
        self.main_layout.addWidget(self.final_optimization_minimizer_maximum_iteration_input, 6, 1)

        self.global_optimization_maximum_iteration_input.valueChanged.connect(self.on_settings_changed)
        self.global_optimization_success_iteration_input.valueChanged.connect(self.on_settings_changed)
        self.global_optimization_step_size_input.valueChanged.connect(self.on_settings_changed)
        self.global_optimization_minimizer_tolerance_level_input.valueChanged.connect(self.on_settings_changed)
        self.global_optimization_minimizer_maximum_iteration_input.valueChanged.connect(self.on_settings_changed)
        self.final_optimization_minimizer_tolerance_level_input.valueChanged.connect(self.on_settings_changed)
        self.final_optimization_minimizer_maximum_iteration_input.valueChanged.connect(self.on_settings_changed)

    @property
    def algorithm_settings(self):
        global_optimization_maximum_iteration = self.global_optimization_maximum_iteration_input.value()
        global_optimization_success_iteration = self.global_optimization_success_iteration_input.value()
        global_optimization_step_size = self.global_optimization_step_size_input.value()
        global_optimization_minimizer_tolerance_level = self.global_optimization_minimizer_tolerance_level_input.value()
        global_optimization_minimizer_maximum_iteration = self.global_optimization_minimizer_maximum_iteration_input.value()
        final_optimization_minimizer_tolerance_level = self.final_optimization_minimizer_tolerance_level_input.value()
        final_optimization_minimizer_maximum_iteration = self.final_optimization_minimizer_maximum_iteration_input.value()

        algorithm_settings = AlgorithmSettings(
            global_optimization_maximum_iteration,
            global_optimization_success_iteration,
            global_optimization_step_size,
            global_optimization_minimizer_tolerance_level,
            global_optimization_minimizer_maximum_iteration,
            final_optimization_minimizer_tolerance_level,
            final_optimization_minimizer_maximum_iteration)
        return algorithm_settings

    def save(self):
        settings = self.algorithm_settings
        if self.setting_file is not None:
            self.setting_file.setValue("global_optimization_maximum_iteration", settings.global_optimization_maximum_iteration)
            self.setting_file.setValue("global_optimization_success_iteration", settings.global_optimization_success_iteration)
            self.setting_file.setValue("global_optimization_step_size", settings.global_optimization_step_size)
            self.setting_file.setValue("global_optimization_minimizer_tolerance_level", settings.global_optimization_minimizer_tolerance_level)
            self.setting_file.setValue("global_optimization_minimizer_maximum_iteration", settings.global_optimization_minimizer_maximum_iteration)
            self.setting_file.setValue("final_optimization_minimizer_tolerance_level", settings.final_optimization_minimizer_tolerance_level)
            self.setting_file.setValue("final_optimization_minimizer_maximum_iteration", settings.final_optimization_minimizer_maximum_iteration)

            self.logger.info("Algorithm settings have been saved to the file.")


    def restore(self):
        if self.setting_file is not None:
            self.global_optimization_maximum_iteration_input.setValue(self.setting_file.value("global_optimization_maximum_iteration", defaultValue=100, type=int))
            self.global_optimization_success_iteration_input.setValue(self.setting_file.value("global_optimization_success_iteration", defaultValue=3, type=int))
            self.global_optimization_step_size_input.setValue(self.setting_file.value("global_optimization_step_size", defaultValue=1.0, type=float))
            self.global_optimization_minimizer_tolerance_level_input.setValue(self.setting_file.value("global_optimization_minimizer_tolerance_level", defaultValue=8, type=int))
            self.global_optimization_minimizer_maximum_iteration_input.setValue(self.setting_file.value("global_optimization_minimizer_maximum_iteration", defaultValue=500, type=int))
            self.final_optimization_minimizer_tolerance_level_input.setValue(self.setting_file.value("final_optimization_minimizer_tolerance_level", defaultValue=100, type=int))
            self.final_optimization_minimizer_maximum_iteration_input.setValue(self.setting_file.value("final_optimization_minimizer_maximum_iteration", defaultValue=1000, type=int))

            self.logger.info("Algorithm settings have been retored from the file.")


    def on_settings_changed(self):
        self.setting_changed_signal.emit(self.algorithm_settings)
        self.logger.info("Algorithm settings have been changed.")


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main = AlgorithmSettingWidget()
    main.show()
    sys.exit(app.exec_())
