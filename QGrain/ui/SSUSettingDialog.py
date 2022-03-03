__all__ = ["SSUSettingDialog"]

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..ssu import SSUAlgorithmSetting, built_in_distances, built_in_minimizers


class SSUSettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=QtCore.Qt.Window)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("SSU Settings"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        # distance
        self.distance_label = QtWidgets.QLabel(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The function to calculate the difference (on the contrary, similarity) between two samples."))
        self.distance_combo_box = QtWidgets.QComboBox()
        self.distance_combo_box.addItems(built_in_distances)
        self.distance_combo_box.setCurrentText("log10MSE")
        self.main_layout.addWidget(self.distance_label, 0, 0)
        self.main_layout.addWidget(self.distance_combo_box, 0, 1)
        # minimizer
        self.minimizer_label = QtWidgets.QLabel(self.tr("Minimizer"))
        self.minimizer_label.setToolTip(self.tr("The minimizer to find the minimum value of the distance function."))
        self.minimizer_combo_box = QtWidgets.QComboBox()
        self.minimizer_combo_box.addItems(built_in_minimizers)
        self.minimizer_combo_box.setCurrentText("SLSQP")
        self.main_layout.addWidget(self.minimizer_label, 1, 0)
        self.main_layout.addWidget(self.minimizer_combo_box, 1, 1)
        # try_GO
        self.try_GO_checkbox = QtWidgets.QCheckBox(self.tr("Global Optimization"))
        self.try_GO_checkbox.setChecked(False)
        self.try_GO_checkbox.setToolTip(self.tr("Whether to try global optimization (GO) before local optimization (LO)."))
        self.main_layout.addWidget(self.try_GO_checkbox, 2, 0, 1, 2)
        # GO_max_niter
        self.GO_max_niter_label = QtWidgets.QLabel(self.tr("[GO] Maximum Number of Iterations"))
        self.GO_max_niter_label.setToolTip(self.tr("Maximum number of iterations of global optimization for termination."))
        self.GO_max_niter_input = QtWidgets.QSpinBox()
        self.GO_max_niter_input.setRange(0, 10000)
        self.GO_max_niter_input.setValue(100)
        self.main_layout.addWidget(self.GO_max_niter_label, 3, 0)
        self.main_layout.addWidget(self.GO_max_niter_input, 3, 1)
        # GO_success_niter
        self.GO_success_niter_label = QtWidgets.QLabel(self.tr("[GO] Success Number of Iterations"))
        self.GO_success_niter_label.setToolTip(self.tr("The number of iteration that reaching the same local optimal value for termination."))
        self.GO_success_niter_input = QtWidgets.QSpinBox()
        self.GO_success_niter_input.setRange(1, 100)
        self.GO_success_niter_input.setValue(5)
        self.main_layout.addWidget(self.GO_success_niter_label, 4, 0)
        self.main_layout.addWidget(self.GO_success_niter_input, 4, 1)
        # GO_step
        self.GO_step_label = QtWidgets.QLabel(self.tr("[GO] Step Size"))
        self.GO_step_label.setToolTip(self.tr("The step size of searching global optimal value."))
        self.GO_step_input = QtWidgets.QDoubleSpinBox()
        self.GO_step_input.setRange(0.01, 10)
        self.GO_step_input.setValue(0.1)
        self.main_layout.addWidget(self.GO_step_label, 5, 0)
        self.main_layout.addWidget(self.GO_step_input, 5, 1)
        # GO_minimizer_max_niter
        self.GO_minimizer_max_niter_label = QtWidgets.QLabel(self.tr("[GO] Minimizer Maximum Number of Iterations"))
        self.GO_minimizer_max_niter_label.setToolTip(self.tr("Maximum number of iterations of the minimizer of global optimization."))
        self.GO_minimizer_max_niter_input = QtWidgets.QSpinBox()
        self.GO_minimizer_max_niter_input.setRange(0, 100000)
        self.GO_minimizer_max_niter_input.setValue(500)
        self.main_layout.addWidget(self.GO_minimizer_max_niter_label, 6, 0)
        self.main_layout.addWidget(self.GO_minimizer_max_niter_input, 6, 1)
        # LO_max_niter
        self.LO_max_niter_label = QtWidgets.QLabel(self.tr("[LO] Maximum Number of Iterations"))
        self.LO_max_niter_label.setToolTip(self.tr("Maximum number of iterations of the minimizer of final local optimization."))
        self.LO_max_niter_input = QtWidgets.QSpinBox()
        self.LO_max_niter_input.setRange(100, 100000)
        self.LO_max_niter_input.setValue(1000)
        self.main_layout.addWidget(self.LO_max_niter_label, 7, 0)
        self.main_layout.addWidget(self.LO_max_niter_input, 7, 1)

    @property
    def setting(self) -> SSUAlgorithmSetting:
        setting = SSUAlgorithmSetting(
            distance=self.distance_combo_box.currentText(),
            minimizer=self.minimizer_combo_box.currentText(),
            try_GO=self.try_GO_checkbox.isChecked(),
            GO_max_niter=self.GO_max_niter_input.value(),
            GO_success_niter=self.GO_success_niter_input.value(),
            GO_step=self.GO_step_input.value(),
            GO_minimizer_max_niter=self.GO_minimizer_max_niter_input.value(),
            LO_max_niter=self.LO_max_niter_input.value())
        return setting

    @setting.setter
    def setting(self, setting: SSUAlgorithmSetting):
        self.distance_combo_box.setCurrentText(setting.distance)
        self.minimizer_combo_box.setCurrentText(setting.minimizer)
        self.try_GO_checkbox.setChecked(setting.try_GO)
        self.GO_max_niter_input.setValue(setting.GO_max_niter)
        self.GO_success_niter_input.setValue(setting.GO_success_niter)
        self.GO_step_input.setValue(setting.GO_step)
        self.GO_minimizer_max_niter_input.setValue(setting.GO_minimizer_max_niter)
        self.LO_max_niter_input.setValue(setting.LO_max_niter)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("SSU Settings"))
        self.distance_label.setText(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The function to calculate the difference (on the contrary, similarity) between two samples."))
        self.minimizer_label.setText(self.tr("Minimizer"))
        self.minimizer_label.setToolTip(self.tr("The minimizer to find the minimum value of the distance function."))
        self.try_GO_checkbox.setText(self.tr("Global Optimization"))
        self.try_GO_checkbox.setToolTip(self.tr("Whether to try global optimization (GO) before local optimization (LO)."))
        self.GO_max_niter_label.setText(self.tr("[GO] Maximum Number of Iterations"))
        self.GO_max_niter_label.setToolTip(self.tr("Maximum number of iterations of global optimization for termination."))
        self.GO_success_niter_label.setText(self.tr("[GO] Success Number of Iterations"))
        self.GO_success_niter_label.setToolTip(self.tr("The number of iteration that reaching the same local optimal value for termination."))
        self.GO_step_label.setText(self.tr("[GO] Step Size"))
        self.GO_step_label.setToolTip(self.tr("The step size of searching global optimal value."))
        self.GO_minimizer_max_niter_label.setText(self.tr("[GO] Minimizer Maximum Number of Iterations"))
        self.GO_minimizer_max_niter_label.setToolTip(self.tr("Maximum number of iterations of the minimizer of global optimization."))
        self.LO_max_niter_label.setText(self.tr("[LO] Maximum Number of Iterations"))
        self.LO_max_niter_label.setToolTip(self.tr("Maximum number of iterations of the minimizer of final local optimization."))
