__all__ = ["ClassicResolverSettingWidget"]

import pickle

import numpy as np
from PySide2.QtCore import QSettings, Qt
from PySide2.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGridLayout, QLabel, QSpinBox, QDialog)
from QGrain.models.ClassicResolverSetting import (ClassicResolverSetting,
                                                  built_in_distances,
                                                  built_in_minimizers)


class ClassicResolverSettingWidget(QDialog):
    def __init__(self, parent=None, filename=None, group=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("Classic Resolver Setting"))
        if filename is not None:
            self.setting_file = QSettings(filename, QSettings.Format.IniFormat)
            if group is not None:
                self.setting_file.beginGroup(group)
        else:
            self.setting_file = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        # distance
        self.distance_label = QLabel(self.tr("Distance (Loss) Function"))
        self.distance_label.setToolTip(self.tr("It's the function to calculate the difference (on the contrary, similarity) between two samples."))
        self.distance_combo_box = QComboBox()
        self.distance_combo_box.addItems(built_in_distances)
        self.distance_combo_box.setCurrentText("log10MSE")
        self.distance_combo_box.currentTextChanged.connect(self.on_distance_changed)
        self.main_layout.addWidget(self.distance_label, 0, 0)
        self.main_layout.addWidget(self.distance_combo_box, 0, 1)
        # minimizer
        self.minimizer_label = QLabel(self.tr("Minimizer"))
        self.minimizer_label.setToolTip(self.tr("The algorithm to find the minimum value of the distance function."))
        self.minimizer_combo_box = QComboBox()
        self.minimizer_combo_box.addItems(built_in_minimizers)
        self.minimizer_combo_box.setCurrentText("SLSQP")
        self.main_layout.addWidget(self.minimizer_label, 1, 0)
        self.main_layout.addWidget(self.minimizer_combo_box, 1, 1)
        # try_GO
        self.try_GO_checkbox = QCheckBox(self.tr("Try Global Optimization (GO)"))
        self.try_GO_checkbox.setChecked(False)
        self.try_GO_checkbox.setToolTip(self.tr("Whether to try global optimization."))
        self.main_layout.addWidget(self.try_GO_checkbox, 2, 0, 1, 2)
        # GO_max_niter
        self.GO_max_niter_label = QLabel(self.tr("[GO] Maximum N<sub>iteration</sub>"))
        self.GO_max_niter_label.setToolTip(self.tr("Maximum number of iterations of global optimization for termination."))
        self.GO_max_niter_input = QSpinBox()
        self.GO_max_niter_input.setRange(0, 10000)
        self.GO_max_niter_input.setValue(100)
        self.main_layout.addWidget(self.GO_max_niter_label, 3, 0)
        self.main_layout.addWidget(self.GO_max_niter_input, 3, 1)
        # GO_success_niter
        self.GO_success_niter_label = QLabel(self.tr("[GO] Success N<sub>iteration</sub>"))
        self.GO_success_niter_label.setToolTip(self.tr("The number of iteration that reaching the same local optimal value for termination."))
        self.GO_success_niter_input = QSpinBox()
        self.GO_success_niter_input.setRange(1, 100)
        self.GO_success_niter_input.setValue(5)
        self.main_layout.addWidget(self.GO_success_niter_label, 4, 0)
        self.main_layout.addWidget(self.GO_success_niter_input, 4, 1)
        # GO_step
        self.GO_step_label = QLabel(self.tr("[GO] Step Size"))
        self.GO_step_label.setToolTip(self.tr("The step size of searching global optimal value."))
        self.GO_step_input = QDoubleSpinBox()
        self.GO_step_input.setRange(0.01, 10)
        self.GO_step_input.setValue(0.1)
        self.main_layout.addWidget(self.GO_step_label, 5, 0)
        self.main_layout.addWidget(self.GO_step_input, 5, 1)
        # GO_minimizer_tol
        self.GO_minimizer_tol_label = QLabel(self.tr("[GO] Minimizer -lg(loss<sub>tolerance</sub>)"))
        self.GO_minimizer_tol_label.setToolTip(self.tr("Controls the tolerance of the loss function for termination."))
        self.GO_minimizer_tol_input = QSpinBox()
        self.GO_minimizer_tol_input.setRange(1, 100)
        self.GO_minimizer_tol_input.setValue(6)
        self.main_layout.addWidget(self.GO_minimizer_tol_label, 6, 0)
        self.main_layout.addWidget(self.GO_minimizer_tol_input, 6, 1)
        # GO_minimizer_ftol
        self.GO_minimizer_ftol_label = QLabel(self.tr("[GO] Minimizer -lg(δ<sub>loss</sub>)"))
        self.GO_minimizer_ftol_label.setToolTip(self.tr("Controls the precision goal for the value of loss function in the stopping criterion."))
        self.GO_minimizer_ftol_input = QSpinBox()
        self.GO_minimizer_ftol_input.setRange(1, 100)
        self.GO_minimizer_ftol_input.setValue(8)
        self.main_layout.addWidget(self.GO_minimizer_ftol_label, 7, 0)
        self.main_layout.addWidget(self.GO_minimizer_ftol_input, 7, 1)
        # GO_minimizer_max_niter
        self.GO_minimizer_max_niter_label = QLabel(self.tr("[GO] Minimizer Maximum N<sub>iteration</sub>"))
        self.GO_minimizer_max_niter_label.setToolTip(self.tr("Maximum number of iterations of the minimizer of global optimization."))
        self.GO_minimizer_max_niter_input = QSpinBox()
        self.GO_minimizer_max_niter_input.setRange(0, 100000)
        self.GO_minimizer_max_niter_input.setValue(500)
        self.main_layout.addWidget(self.GO_minimizer_max_niter_label, 8, 0)
        self.main_layout.addWidget(self.GO_minimizer_max_niter_input, 8, 1)
        # FLO_tol
        self.FLO_tol_label = QLabel(self.tr("[FLO] -lg(loss<sub>tolerance</sub>)"))
        self.FLO_tol_label.setToolTip(self.tr("Controls the tolerance of the loss function for termination."))
        self.FLO_tol_input = QSpinBox()
        self.FLO_tol_input.setRange(1, 100)
        self.FLO_tol_input.setValue(8)
        self.main_layout.addWidget(self.FLO_tol_label, 9, 0)
        self.main_layout.addWidget(self.FLO_tol_input, 9, 1)
        # FLO_ftol
        self.FLO_ftol_label = QLabel(self.tr("[FLO] -lg(δ<sub>loss</sub>)"))
        self.FLO_ftol_label.setToolTip(self.tr("Controls the precision goal for the value of loss function in the stopping criterion."))
        self.FLO_ftol_input = QSpinBox()
        self.FLO_ftol_input.setRange(1, 100)
        self.FLO_ftol_input.setValue(10)
        self.main_layout.addWidget(self.FLO_ftol_label, 10, 0)
        self.main_layout.addWidget(self.FLO_ftol_input, 10, 1)
        # FLO_max_niter
        self.FLO_max_niter_label = QLabel(self.tr("[FLO] Maximum N<sub>iteration</sub>"))
        self.FLO_max_niter_label.setToolTip(self.tr("Maximum number of iterations of the minimizer of final local optimization."))
        self.FLO_max_niter_input = QSpinBox()
        self.FLO_max_niter_input.setRange(100, 100000)
        self.FLO_max_niter_input.setValue(1000)
        self.main_layout.addWidget(self.FLO_max_niter_label, 11, 0)
        self.main_layout.addWidget(self.FLO_max_niter_input, 11, 1)

    def on_distance_changed(self, distance: str):
        if distance == "log10MSE":
            self.GO_minimizer_tol_label.setText(self.tr("[GO] Minimizer -loss<sub>tolerance</sub>"))
            self.FLO_tol_label.setText(self.tr("[FLO] -loss<sub>tolerance</sub>"))
        else:
            self.GO_minimizer_tol_label.setText(self.tr("[GO] Minimizer -lg(loss<sub>tolerance</sub>)"))
            self.FLO_tol_label.setText(self.tr("[FLO] -lg(loss<sub>tolerance</sub>)"))

    @property
    def setting(self):
        distance = self.distance_combo_box.currentText()
        minimizer = self.minimizer_combo_box.currentText()
        try_GO = self.try_GO_checkbox.isChecked()
        GO_max_niter = self.GO_max_niter_input.value()
        GO_success_niter = self.GO_success_niter_input.value()
        GO_step_size = self.GO_step_input.value()
        GO_minimizer_tol = -self.GO_minimizer_tol_input.value() if distance == "log10MSE" else 10**(-self.GO_minimizer_tol_input.value())
        GO_minimizer_ftol = 10**(-self.GO_minimizer_ftol_input.value())
        GO_minimizer_max_niter = self.GO_minimizer_max_niter_input.value()
        FLO_tol = -self.FLO_tol_input.value() if distance == "log10MSE" else 10**(-self.FLO_tol_input.value())
        FLO_ftol = 10**(-self.FLO_ftol_input.value())
        FLO_max_niter = self.FLO_max_niter_input.value()

        setting = ClassicResolverSetting(
            distance=distance,
            minimizer=minimizer,
            try_GO=try_GO,
            GO_max_niter=GO_max_niter,
            GO_success_niter=GO_success_niter,
            GO_step=GO_step_size,
            GO_minimizer_tol=GO_minimizer_tol,
            GO_minimizer_ftol=GO_minimizer_ftol,
            GO_minimizer_max_niter=GO_minimizer_max_niter,
            FLO_tol=FLO_tol,
            FLO_ftol=FLO_ftol,
            FLO_max_niter=FLO_max_niter)
        return setting

    @setting.setter
    def setting(self, setting: ClassicResolverSetting):
        self.distance_combo_box.setCurrentText(setting.distance)
        self.minimizer_combo_box.setCurrentText(setting.minimizer)
        self.try_GO_checkbox.setChecked(setting.try_GO)
        self.GO_max_niter_input.setValue(setting.GO_max_niter)
        self.GO_success_niter_input.setValue(setting.GO_success_niter)
        self.GO_step_input.setValue(setting.GO_step)
        if setting.distance == "log10MSE":
            self.GO_minimizer_tol_input.setValue(-setting.GO_minimizer_tol)
            self.FLO_tol_input.setValue(-setting.FLO_tol)
        else:
            self.GO_minimizer_tol_input.setValue(-np.log10(setting.GO_minimizer_tol))
            self.FLO_ftol_input.setValue(-np.log10(setting.FLO_tol))
        self.GO_minimizer_ftol_input.setValue(-np.log10(setting.GO_minimizer_ftol))
        self.GO_minimizer_max_niter_input.setValue(setting.GO_minimizer_max_niter)
        self.FLO_ftol_input.setValue(-np.log10(setting.FLO_ftol))
        self.FLO_max_niter_input.setValue(setting.FLO_max_niter)

    def save(self):
        if self.setting_file is not None:
            setting_bytes = pickle.dumps(self.setting)
            self.setting_file.setValue("classic_resolver_setting", setting_bytes)

    def restore(self):
        if self.setting_file is not None:
            setting_bytes = self.setting_file.value("classic_resolver_setting", defaultValue=None)
            if setting_bytes is not None:
                setting = pickle.loads(setting_bytes)
                self.setting = setting
        else:
            self.setting = ClassicResolverSetting()

if __name__ == "__main__":
    import sys
    from QGrain.entry import setup_app
    app, splash = setup_app()
    main = ClassicResolverSettingWidget()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
