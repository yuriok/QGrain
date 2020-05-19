__all__ = ["AlgorithmSetting"]

import logging
import os

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtWidgets import (QGridLayout, QLabel, QSpinBox, QDoubleSpinBox, QMessageBox,
                               QWidget)

QGRAIN_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


class AlgorithmSetting(QWidget):
    sigAlgorithmSettingChanged = Signal(dict)
    logger = logging.getLogger("root.ui.AlgorithmSetting")
    gui_logger = logging.getLogger("GUI")
    def __init__(self):
        super().__init__()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.settings = QSettings(os.path.join(QGRAIN_ROOT_PATH, "settings", "QGrain.ini"), QSettings.Format.IniFormat)
        self.settings.beginGroup("algorithm")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)

        self.title_label = QLabel(self.tr("Algorithm Settings:"))
        self.title_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.title_label, 0, 0)

        self.global_maxiter_label = QLabel(self.tr("Global Optimization Max Iteration"))
        self.global_maxiter_label.setToolTip(self.tr("Max iteration number of global optimization.\nIf the global optimization iteration has reached the max number, fitting process will stop."))
        self.main_layout.addWidget(self.global_maxiter_label, 1, 0)
        self.global_maxiter_input = QSpinBox()
        self.global_maxiter_input.setRange(0, 10000)
        self.global_maxiter_input.setValue(100)
        self.main_layout.addWidget(self.global_maxiter_input, 1, 1)

        self.global_success_iter_label = QLabel(self.tr("Global Optimization Success Iteration"))
        self.global_success_iter_label.setToolTip(self.tr("It's one of the terminal conditions of global optimization.\nIt means the iteration number of reaching the same minimum."))
        self.main_layout.addWidget(self.global_success_iter_label, 2, 0)
        self.global_success_iter_input = QSpinBox()
        self.global_success_iter_input.setRange(0, 100)
        self.global_success_iter_input.setValue(3)
        self.main_layout.addWidget(self.global_success_iter_input, 2, 1)

        self.global_stepsize_label = QLabel(self.tr("Global Optimization Stepsize"))
        self.global_stepsize_label.setToolTip(self.tr("The stepsize of searching global minimum.\nGreater stepsize will jump out the local minimum easier but may miss the global minimum."))
        self.main_layout.addWidget(self.global_stepsize_label, 3, 0)
        self.global_stepsize_input = QDoubleSpinBox()
        self.global_stepsize_input.setRange(0.0001, 10000)
        self.global_stepsize_input.setValue(1.0)
        self.main_layout.addWidget(self.global_stepsize_input, 3, 1)

        self.minimizer_tolerance_level_label = QLabel(self.tr("Global Minimizer Tolerance Level"))
        self.minimizer_tolerance_level_label.setToolTip(self.tr("The tolerance level of the minimizer of global optimization.\nTolerance level means the accepted minimum variation (10 ^ -level) of the target function.\nIt controls the precision and speed of fitting.\nIt's recommended to use ralatively lower level in global optimization process but higher leverl in final fitting."))
        self.main_layout.addWidget(self.minimizer_tolerance_level_label, 4, 0)
        self.minimizer_tolerance_level_input = QSpinBox()
        self.minimizer_tolerance_level_input.setRange(1, 1000)
        self.minimizer_tolerance_level_input.setValue(8)
        self.main_layout.addWidget(self.minimizer_tolerance_level_input, 4, 1)

        self.minimizer_maxiter_label = QLabel(self.tr("Global Minimizer Max Iteration"))
        self.minimizer_maxiter_label.setToolTip(self.tr("Max iteration number of the minimizer of global optimization."))
        self.main_layout.addWidget(self.minimizer_maxiter_label, 5, 0)
        self.minimizer_maxiter_input = QSpinBox()
        self.minimizer_maxiter_input.setRange(0, 10000)
        self.minimizer_maxiter_input.setValue(200)
        self.main_layout.addWidget(self.minimizer_maxiter_input, 5, 1)

        self.final_tolerance_level_label = QLabel(self.tr("Final Fitting Tolerance Level"))
        self.final_tolerance_level_label.setToolTip(self.tr("The tolerance level of the minimizer of final fitting."))
        self.main_layout.addWidget(self.final_tolerance_level_label, 6, 0)
        self.final_tolerance_level_input = QSpinBox()
        self.final_tolerance_level_input.setRange(1, 1000)
        self.final_tolerance_level_input.setValue(100)
        self.main_layout.addWidget(self.final_tolerance_level_input, 6, 1)

        self.final_maxiter_label = QLabel(self.tr("Final Fitting Max Iteration"))
        self.final_maxiter_label.setToolTip(self.tr("Max iteration number of the minimizer of final fitting."))
        self.main_layout.addWidget(self.final_maxiter_label, 7, 0)
        self.final_maxiter_input = QSpinBox()
        self.final_maxiter_input.setRange(100, 10000)
        self.final_maxiter_input.setValue(1000)
        self.main_layout.addWidget(self.final_maxiter_input, 7, 1)

        self.global_maxiter_input.valueChanged.connect(self.on_settings_changed)
        self.global_success_iter_input.valueChanged.connect(self.on_settings_changed)
        self.global_stepsize_input.valueChanged.connect(self.on_settings_changed)
        self.minimizer_tolerance_level_input.valueChanged.connect(self.on_settings_changed)
        self.minimizer_maxiter_input.valueChanged.connect(self.on_settings_changed)
        self.final_tolerance_level_input.valueChanged.connect(self.on_settings_changed)
        self.final_maxiter_input.valueChanged.connect(self.on_settings_changed)

    def on_settings_changed(self):
        global_optimization_maxiter = self.global_maxiter_input.value()
        global_optimization_success_iter = self.global_success_iter_input.value()
        global_optimization_stepsize = self.global_stepsize_input.value()
        minimizer_tolerance_level = self.minimizer_tolerance_level_input.value()
        minimizer_maxiter = self.minimizer_maxiter_input.value()
        final_tolerance_level = self.final_tolerance_level_input.value()
        final_maxiter = self.final_maxiter_input.value()

        signal_data = dict(
            global_optimization_maxiter=global_optimization_maxiter,
            global_optimization_success_iter=global_optimization_success_iter,
            global_optimization_stepsize=global_optimization_stepsize,
            minimizer_tolerance=10**(-minimizer_tolerance_level),
            minimizer_maxiter=minimizer_maxiter,
            final_tolerance=10**(-final_tolerance_level),
            final_maxiter=final_maxiter)
        self.sigAlgorithmSettingChanged.emit(signal_data)

        self.settings.setValue("global_optimization_maxiter", global_optimization_maxiter)
        self.settings.setValue("global_optimization_success_iter", global_optimization_success_iter)
        self.settings.setValue("global_optimization_stepsize", global_optimization_stepsize)
        self.settings.setValue("minimizer_tolerance_level", minimizer_tolerance_level)
        self.settings.setValue("minimizer_maxiter", minimizer_maxiter)
        self.settings.setValue("final_tolerance_level", final_tolerance_level)
        self.settings.setValue("final_maxiter", final_maxiter)

    def restore(self):
        self.global_maxiter_input.setValue(self.settings.value("global_optimization_maxiter", defaultValue=100, type=int))
        self.global_success_iter_input.setValue(self.settings.value("global_optimization_success_iter", defaultValue=3, type=int))
        self.global_stepsize_input.setValue(self.settings.value("global_optimization_stepsize", defaultValue=1.0, type=float))
        self.minimizer_tolerance_level_input.setValue(self.settings.value("minimizer_tolerance_level", defaultValue=8, type=int))
        self.minimizer_maxiter_input.setValue(self.settings.value("minimizer_maxiter", defaultValue=500, type=int))
        self.final_tolerance_level_input.setValue(self.settings.value("final_tolerance_level", defaultValue=100, type=int))
        self.final_maxiter_input.setValue(self.settings.value("final_maxiter", defaultValue=1000, type=int))


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main = AlgorithmSetting()
    main.show()
    sys.exit(app.exec_())
