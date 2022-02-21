__all__ = ["EMMASettingDialog"]

import numpy as np
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QGridLayout,
                               QLabel, QSpinBox)

from ..emma import EMMAResolverSetting, built_in_distances


class EMMASettingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("EMMA Resolver Setting"))
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.initialize_ui()

    def initialize_ui(self):
        self.main_layout = QGridLayout(self)
        self.device_label = QLabel(self.tr("Device"))
        self.device_label.setToolTip(self.tr("The neural netowrk framwork, pytorch, also can use the GPU of Nvidia to do calculations."))
        self.device_combo_box = QComboBox()
        self.device_combo_box.addItem("cpu")
        if torch.cuda.is_available():
            self.device_combo_box.addItem("cuda")
            self.device_combo_box.setCurrentText("cuda")
        self.main_layout.addWidget(self.device_label, 0, 0)
        self.main_layout.addWidget(self.device_combo_box, 0, 1)

        self.distance_label = QLabel(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The function to calculate the difference (on the contrary, similarity) between two samples."))
        self.distance_combo_box = QComboBox()
        self.distance_combo_box.addItems(built_in_distances)
        self.distance_combo_box.setCurrentText("log10MSE")
        self.distance_combo_box.currentTextChanged.connect(self.on_distance_changed)
        self.main_layout.addWidget(self.distance_label, 1, 0)
        self.main_layout.addWidget(self.distance_combo_box, 1, 1)

        self.min_niter_label = QLabel(self.tr("Minimum Number of Iterations"))
        self.min_niter_label.setToolTip(self.tr("Minimum number of iterations to perform"))
        self.min_niter_input = QSpinBox()
        self.min_niter_input.setRange(10, 10000)
        self.min_niter_input.setValue(2000)
        self.main_layout.addWidget(self.min_niter_label, 2, 0)
        self.main_layout.addWidget(self.min_niter_input, 2, 1)

        self.max_niter_label = QLabel(self.tr("Maximum Number of Iterations"))
        self.max_niter_label.setToolTip(self.tr("Maximum number of iterations to perform"))
        self.max_niter_input = QSpinBox()
        self.max_niter_input.setRange(100, 100000)
        self.max_niter_input.setValue(5000)
        self.main_layout.addWidget(self.max_niter_label, 3, 0)
        self.main_layout.addWidget(self.max_niter_input, 3, 1)

        self.tol_label = QLabel(self.tr("-lg(loss<sub>tolerance</sub>)"))
        self.tol_label.setToolTip(self.tr("Controls the tolerance of the loss function for termination."))
        self.tol_input = QSpinBox()
        self.tol_input.setRange(1, 100)
        self.tol_input.setValue(10)
        self.main_layout.addWidget(self.tol_label, 4, 0)
        self.main_layout.addWidget(self.tol_input, 4, 1)

        self.ftol_label = QLabel(self.tr("-lg(δ<sub>loss</sub>)"))
        self.ftol_label.setToolTip(self.tr("Controls the precision goal for the value of loss function in the stopping criterion."))
        self.ftol_input = QSpinBox()
        self.ftol_input.setRange(1, 100)
        self.ftol_input.setValue(10)
        self.main_layout.addWidget(self.ftol_label, 5, 0)
        self.main_layout.addWidget(self.ftol_input, 5, 1)

        self.lr_label = QLabel(self.tr("Learning Rate (x10<sup>-3</sup>)"))
        self.lr_label.setToolTip(self.tr("The learning rate of the neural network to update its weights from gradient."))
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(3)
        self.lr_input.setRange(0.001, 1000)
        self.lr_input.setValue(15)
        self.main_layout.addWidget(self.lr_label, 6, 0)
        self.main_layout.addWidget(self.lr_input, 6, 1)


    def on_distance_changed(self, distance: str):
        if distance == "log10MSE":
            self.tol_label.setText(self.tr("-loss<sub>tolerance</sub>"))
        else:
            self.tol_label.setText(self.tr("-lg(loss<sub>tolerance</sub>)"))

    @property
    def setting(self):
        devices = ["cpu", "cuda"]
        device = devices[self.device_combo_box.currentIndex()]
        distance = self.distance_combo_box.currentText()
        min_niter = self.min_niter_input.value()
        max_niter = self.max_niter_input.value()
        # when using Lg(MSE) distance
        tol = -self.tol_input.value() if distance == "log10MSE" else 10**(-self.tol_input.value())
        ftol = 10**(-self.ftol_input.value())
        lr = self.lr_input.value() / 1000.0

        setting = EMMAResolverSetting(device=device, distance=distance,
                                    min_niter=min_niter, max_niter=max_niter,
                                    tol=tol, ftol=ftol, lr=lr)
        return setting

    @setting.setter
    def setting(self, setting: EMMAResolverSetting):
        self.device_combo_box.setCurrentText(setting.device)
        self.distance_combo_box.setCurrentText(setting.distance)
        self.min_niter_input.setValue(setting.min_niter)
        self.max_niter_input.setValue(setting.max_niter)
        if setting.distance == "log10MSE":
            self.tol_input.setValue(-setting.tol)
        else:
            self.tol_input.setValue(-np.log10(setting.tol))
        self.ftol_input.setValue(-np.log10(setting.ftol))
        self.lr_input.setValue(setting.lr*1000.0)
