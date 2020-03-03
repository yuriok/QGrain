__all__ = ["AlgorithmSetting"]

import logging
import os

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtGui import QDoubleValidator, QIntValidator, QValidator
from PySide2.QtWidgets import (QGridLayout, QLabel, QLineEdit, QMessageBox,
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
        self.int_validator = QIntValidator()
        self.int_validator.setBottom(0)
        self.double_validator = QDoubleValidator()
        self.double_validator.setBottom(0.0)

        self.title_label = QLabel(self.tr("Algorithm Settings:"))
        self.title_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.title_label, 0, 0)

        self.global_maxiter_label = QLabel(self.tr("Global Optimization Max Iteration"))
        self.global_maxiter_label.setToolTip(self.tr("Max iteration number of global optimization.\nIf the global optimization iteration has reached the max number, fitting process will stop."))
        self.main_layout.addWidget(self.global_maxiter_label, 1, 0)
        self.global_maxiter_edit = QLineEdit()
        self.global_maxiter_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.global_maxiter_edit, 1, 1)

        self.global_success_iter_label = QLabel(self.tr("Global Optimization Success Iteration"))
        self.global_success_iter_label.setToolTip(self.tr("It's one of the terminal conditions of global optimization.\nIt means the iteration number of reaching the same minimum."))
        self.main_layout.addWidget(self.global_success_iter_label, 2, 0)
        self.global_success_iter_edit = QLineEdit()
        self.global_success_iter_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.global_success_iter_edit, 2, 1)

        self.global_stepsize_label = QLabel(self.tr("Global Optimization Stepsize"))
        self.global_stepsize_label.setToolTip(self.tr("The stepsize of searching global minimum.\nGreater stepsize will jump out the local minimum easier but may miss the global minimum."))
        self.main_layout.addWidget(self.global_stepsize_label, 3, 0)
        self.global_stepsize_edit = QLineEdit()
        self.global_stepsize_edit.setValidator(self.double_validator)
        self.main_layout.addWidget(self.global_stepsize_edit, 3, 1)

        self.minimizer_tolerance_level_label = QLabel(self.tr("Global Minimizer Tolerance Level"))
        self.minimizer_tolerance_level_label.setToolTip(self.tr("The tolerance level of the minimizer of global optimization.\nTolerance level means the accepted minimum variation (10 ^ -level) of the target function.\nIt controls the precision and speed of fitting.\nIt's recommended to use ralatively lower level in global optimization process but higher leverl in final fitting."))
        self.main_layout.addWidget(self.minimizer_tolerance_level_label, 4, 0)
        self.minimizer_tolerance_level_edit = QLineEdit()
        self.minimizer_tolerance_level_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.minimizer_tolerance_level_edit, 4, 1)

        self.minimizer_maxiter_label = QLabel(self.tr("Global Minimizer Max Iteration"))
        self.minimizer_maxiter_label.setToolTip(self.tr("Max iteration number of the minimizer of global optimization."))
        self.main_layout.addWidget(self.minimizer_maxiter_label, 5, 0)
        self.minimizer_maxiter_edit = QLineEdit()
        self.minimizer_maxiter_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.minimizer_maxiter_edit, 5, 1)

        self.final_tolerance_level_label = QLabel(self.tr("Final Fitting Tolerance Level"))
        self.final_tolerance_level_label.setToolTip(self.tr("The tolerance level of the minimizer of final fitting."))
        self.main_layout.addWidget(self.final_tolerance_level_label, 6, 0)
        self.final_tolerance_level_edit = QLineEdit()
        self.final_tolerance_level_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.final_tolerance_level_edit, 6, 1)

        self.final_maxiter_label = QLabel(self.tr("Final Fitting Max Iteration"))
        self.final_maxiter_label.setToolTip(self.tr("Max iteration number of the minimizer of final fitting."))
        self.main_layout.addWidget(self.final_maxiter_label, 7, 0)
        self.final_maxiter_edit = QLineEdit()
        self.final_maxiter_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.final_maxiter_edit, 7, 1)

        self.global_maxiter_edit.textChanged.connect(self.on_settings_changed)
        self.global_success_iter_edit.textChanged.connect(self.on_settings_changed)
        self.global_stepsize_edit.textChanged.connect(self.on_settings_changed)
        self.minimizer_tolerance_level_edit.textChanged.connect(self.on_settings_changed)
        self.minimizer_maxiter_edit.textChanged.connect(self.on_settings_changed)
        self.final_tolerance_level_edit.textChanged.connect(self.on_settings_changed)
        self.final_maxiter_edit.textChanged.connect(self.on_settings_changed)

        self.restore()

    def on_settings_changed(self):
        global_optimization_maxiter = self.global_maxiter_edit.text()
        global_optimization_success_iter = self.global_success_iter_edit.text()
        global_optimization_stepsize = self.global_stepsize_edit.text()
        minimizer_tolerance_level = self.minimizer_tolerance_level_edit.text()
        minimizer_maxiter = self.minimizer_maxiter_edit.text()
        final_tolerance_level = self.final_tolerance_level_edit.text()
        final_maxiter = self.final_maxiter_edit.text()

        try:
            signal_data = dict(
                global_optimization_maxiter=int(global_optimization_maxiter),
                global_optimization_success_iter=int(global_optimization_success_iter),
                global_optimization_stepsize=float(global_optimization_stepsize),
                minimizer_tolerance=10**(-int(minimizer_tolerance_level)),
                minimizer_maxiter=int(minimizer_maxiter),
                final_tolerance=10**(-int(final_tolerance_level)),
                final_maxiter=int(final_maxiter))

            self.sigAlgorithmSettingChanged.emit(signal_data)
        except ValueError:
            return

        self.settings.setValue("global_optimization_maxiter", global_optimization_maxiter)
        self.settings.setValue("global_optimization_success_iter", global_optimization_success_iter)
        self.settings.setValue("global_optimization_stepsize", global_optimization_stepsize)
        self.settings.setValue("minimizer_tolerance_level", minimizer_tolerance_level)
        self.settings.setValue("minimizer_maxiter", minimizer_maxiter)
        self.settings.setValue("final_tolerance_level", final_tolerance_level)
        self.settings.setValue("final_maxiter", final_maxiter)

    def restore(self):
        self.global_maxiter_edit.setText(self.settings.value("global_optimization_maxiter", defaultValue="100"))
        self.global_success_iter_edit.setText(self.settings.value("global_optimization_success_iter", defaultValue="3"))
        self.global_stepsize_edit.setText(self.settings.value("global_optimization_stepsize", defaultValue="1.0"))
        self.minimizer_tolerance_level_edit.setText(self.settings.value("minimizer_tolerance_level", defaultValue="8"))
        self.minimizer_maxiter_edit.setText(self.settings.value("minimizer_maxiter", defaultValue="500"))
        self.final_tolerance_level_edit.setText(self.settings.value("final_tolerance_level", defaultValue="100"))
        self.final_maxiter_edit.setText(self.settings.value("final_maxiter", defaultValue="1000"))
