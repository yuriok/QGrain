__all__ = ["EMMASettingDialog"]

import numpy as np
import torch
from PySide6 import QtCore, QtWidgets

from ..emma import EMMAAlgorithmSetting, built_in_distances


class EMMASettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=QtCore.Qt.Window)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("EMMA Settings"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.device_label = QtWidgets.QLabel(self.tr("Device"))
        self.device_label.setToolTip(self.tr("The neural netowrk framwork, pytorch, also can use the GPU of NVIDIA to do calculations."))
        self.device_combo_box = QtWidgets.QComboBox()
        self.device_combo_box.addItem("CPU")
        if torch.cuda.is_available():
            self.device_combo_box.addItem("CUDA")
            self.device_combo_box.setCurrentText("CUDA")
        self.main_layout.addWidget(self.device_label, 0, 0)
        self.main_layout.addWidget(self.device_combo_box, 0, 1)

        self.distance_label = QtWidgets.QLabel(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The function to calculate the difference (on the contrary, similarity) between two samples."))
        self.distance_combo_box = QtWidgets.QComboBox()
        self.distance_combo_box.addItems(built_in_distances)
        self.distance_combo_box.setCurrentText("log10MSE")
        self.main_layout.addWidget(self.distance_label, 1, 0)
        self.main_layout.addWidget(self.distance_combo_box, 1, 1)

        self.min_epochs_label = QtWidgets.QLabel(self.tr("Minimum Number of Epochs"))
        self.min_epochs_label.setToolTip(self.tr("Minimum number of epochs to be performed."))
        self.min_epochs_input = QtWidgets.QSpinBox()
        self.min_epochs_input.setRange(10, 10000)
        self.min_epochs_input.setValue(2000)
        self.main_layout.addWidget(self.min_epochs_label, 2, 0)
        self.main_layout.addWidget(self.min_epochs_input, 2, 1)

        self.max_epochs_label = QtWidgets.QLabel(self.tr("Maximum Number of Epochs"))
        self.max_epochs_label.setToolTip(self.tr("Maximum number of epochs to be performed."))
        self.max_epochs_input = QtWidgets.QSpinBox()
        self.max_epochs_input.setRange(100, 100000)
        self.max_epochs_input.setValue(5000)
        self.main_layout.addWidget(self.max_epochs_label, 3, 0)
        self.main_layout.addWidget(self.max_epochs_input, 3, 1)

        self.precision_label = QtWidgets.QLabel(self.tr("Precision"))
        self.precision_label.setToolTip(self.tr("It controls the precision for the value of loss function in the stopping criterion."))
        self.precision_input = QtWidgets.QDoubleSpinBox()
        self.precision_input.setRange(2.0, 20.0)
        self.precision_input.setValue(10.0)
        self.main_layout.addWidget(self.precision_label, 5, 0)
        self.main_layout.addWidget(self.precision_input, 5, 1)

        self.learning_rate_label = QtWidgets.QLabel(self.tr("Learning Rate (x10<sup>-3</sup>)"))
        self.learning_rate_label.setToolTip(self.tr("The learning rate of the neural network to update its weights from gradient."))
        self.learning_rate_input = QtWidgets.QDoubleSpinBox()
        self.learning_rate_input.setDecimals(3)
        self.learning_rate_input.setRange(0.001, 1000)
        self.learning_rate_input.setValue(5)
        self.main_layout.addWidget(self.learning_rate_label, 6, 0)
        self.main_layout.addWidget(self.learning_rate_input, 6, 1)

        self.beta1_label = QtWidgets.QLabel(self.tr("Beta 1"))
        self.beta1_label.setToolTip(self.tr("Betas are the coefficients used for computing running averages of gradient and its square."))
        self.beta1_input = QtWidgets.QDoubleSpinBox()
        self.beta1_input.setDecimals(4)
        self.beta1_input.setRange(0.0001, 0.9999)
        self.beta1_input.setValue(0.8000)
        self.main_layout.addWidget(self.beta1_label, 7, 0)
        self.main_layout.addWidget(self.beta1_input, 7, 1)

        self.beta2_label = QtWidgets.QLabel(self.tr("Beta 2"))
        self.beta2_label.setToolTip(self.tr("Betas are the coefficients used for computing running averages of gradient and its square."))
        self.beta2_input = QtWidgets.QDoubleSpinBox()
        self.beta2_input.setDecimals(4)
        self.beta2_input.setRange(0.0001, 0.9999)
        self.beta2_input.setValue(0.5000)
        self.main_layout.addWidget(self.beta2_label, 8, 0)
        self.main_layout.addWidget(self.beta2_input, 8, 1)

    @property
    def setting(self):
        devices = ["cpu", "cuda"]
        setting = EMMAAlgorithmSetting(
            device=devices[self.device_combo_box.currentIndex()],
            distance=self.distance_combo_box.currentText(),
            min_epochs=self.min_epochs_input.value(),
            max_epochs=self.max_epochs_input.value(),
            precision=self.precision_input.value(),
            learning_rate=self.learning_rate_input.value() / 1000.0,
            betas=(self.beta1_input.value(), self.beta2_input.value()))
        return setting

    @setting.setter
    def setting(self, setting: EMMAAlgorithmSetting):
        device_map = {"cpu": 0, "cuda": 1}
        self.device_combo_box.setCurrentIndex(device_map[setting.device])
        self.distance_combo_box.setCurrentText(setting.distance)
        self.min_epochs_input.setValue(setting.min_epochs)
        self.max_epochs_input.setValue(setting.max_epochs)
        self.precision_input.setValue(setting.precision)
        self.learning_rate_input.setValue(setting.learning_rate*1000.0)
        beta1, beta2 = setting.betas
        self.beta1_input.setValue(beta1)
        self.beta2_input.setValue(beta2)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("EMMA Settings"))
        self.device_label.setText(self.tr("Device"))
        self.device_label.setToolTip(self.tr("The neural netowrk framwork, pytorch, also can use the GPU of NVIDIA to do calculations."))
        self.distance_label.setText(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The function to calculate the difference (on the contrary, similarity) between two samples."))
        self.min_epochs_label.setText(self.tr("Minimum Number of Epochs"))
        self.min_epochs_label.setToolTip(self.tr("Minimum number of epochs to be performed."))
        self.max_epochs_label.setText(self.tr("Maximum Number of Epochs"))
        self.max_epochs_label.setToolTip(self.tr("Maximum number of epochs to be performed."))
        self.precision_label.setText(self.tr("Precision"))
        self.precision_label.setToolTip(self.tr("It controls the precision for the value of loss function in the stopping criterion."))
        self.learning_rate_label.setText(self.tr("Learning Rate (x10<sup>-3</sup>)"))
        self.learning_rate_label.setToolTip(self.tr("The learning rate of the neural network to update its weights from gradient."))
        self.beta1_label.setText(self.tr("Beta 1"))
        self.beta1_label.setToolTip(self.tr("Betas are the coefficients used for computing running averages of gradient and its square."))
        self.beta2_label.setText(self.tr("Beta 2"))
        self.beta2_label.setToolTip(self.tr("Betas are the coefficients used for computing running averages of gradient and its square."))
