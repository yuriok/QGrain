__all__ = ["UDMSettings"]

import logging
from typing import *

from PySide6 import QtCore, QtWidgets
from grpc._channel import _InactiveRpcError

from ..protos.client import QGrainClient


class UDMSettings(QtWidgets.QDialog):
    logger = logging.getLogger("QGrain.UDMSettings")

    def __init__(self, client: QGrainClient = None, parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)
        self._client = client
        self.setWindowTitle(self.tr("UDM Settings"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.device_label = QtWidgets.QLabel(self.tr("Device"))
        self.device_label.setToolTip(self.tr(
            "The neural network framework, pytorch, also can use the GPU of NVIDIA to do calculations."))
        self.device_combo_box = QtWidgets.QComboBox()
        self.main_layout.addWidget(self.device_label, 0, 0)
        self.main_layout.addWidget(self.device_combo_box, 0, 1)
        self.pretrain_epochs_label = QtWidgets.QLabel(self.tr("Pretrain Epochs"))
        self.pretrain_epochs_label.setToolTip(self.tr(
            "The number of epochs before formal training. "
            "The pretrain process only update the proportions of end members."))
        self.pretrain_epochs_input = QtWidgets.QSpinBox()
        self.pretrain_epochs_input.setRange(0, 100000)
        self.pretrain_epochs_input.setValue(200)
        self.main_layout.addWidget(self.pretrain_epochs_label, 1, 0)
        self.main_layout.addWidget(self.pretrain_epochs_input, 1, 1)
        self.min_epochs_label = QtWidgets.QLabel(self.tr("Minimum Number of Epochs"))
        self.min_epochs_label.setToolTip(self.tr("Minimum number of epochs to be performed."))
        self.min_epochs_input = QtWidgets.QSpinBox()
        self.min_epochs_input.setRange(10, 10000)
        self.min_epochs_input.setValue(500)
        self.main_layout.addWidget(self.min_epochs_label, 2, 0)
        self.main_layout.addWidget(self.min_epochs_input, 2, 1)
        self.max_epochs_label = QtWidgets.QLabel(self.tr("Maximum Number of Epochs"))
        self.max_epochs_label.setToolTip(self.tr("Maximum number of epochs to be performed."))
        self.max_epochs_input = QtWidgets.QSpinBox()
        self.max_epochs_input.setRange(100, 100000)
        self.max_epochs_input.setValue(1000)
        self.main_layout.addWidget(self.max_epochs_label, 3, 0)
        self.main_layout.addWidget(self.max_epochs_input, 3, 1)
        self.precision_label = QtWidgets.QLabel(self.tr("Precision"))
        self.precision_label.setToolTip(self.tr(
            "It controls the precision for the value of loss function in the stopping criterion."))
        self.precision_input = QtWidgets.QDoubleSpinBox()
        self.precision_input.setRange(2.0, 20.0)
        self.precision_input.setValue(10.0)
        self.main_layout.addWidget(self.precision_label, 4, 0)
        self.main_layout.addWidget(self.precision_input, 4, 1)
        self.learning_rate_label = QtWidgets.QLabel(self.tr("Learning Rate (x10<sup>-3</sup>)"))
        self.learning_rate_label.setToolTip(self.tr(
            "The learning rate of the neural network to update its weights from gradient."))
        self.learning_rate_input = QtWidgets.QDoubleSpinBox()
        self.learning_rate_input.setDecimals(3)
        self.learning_rate_input.setRange(0.001, 1000)
        self.learning_rate_input.setValue(50)
        self.main_layout.addWidget(self.learning_rate_label, 5, 0)
        self.main_layout.addWidget(self.learning_rate_input, 5, 1)
        self.beta1_label = QtWidgets.QLabel(self.tr("Beta 1"))
        self.beta1_label.setToolTip(self.tr(
            "Betas are the coefficients used for computing running averages of gradient and its square."))
        self.beta1_input = QtWidgets.QDoubleSpinBox()
        self.beta1_input.setDecimals(4)
        self.beta1_input.setRange(0.0001, 0.9999)
        self.beta1_input.setValue(0.8000)
        self.main_layout.addWidget(self.beta1_label, 6, 0)
        self.main_layout.addWidget(self.beta1_input, 6, 1)
        self.beta2_label = QtWidgets.QLabel(self.tr("Beta 2"))
        self.beta2_label.setToolTip(self.tr(
            "Betas are the coefficients used for computing running averages of gradient and its square."))
        self.beta2_input = QtWidgets.QDoubleSpinBox()
        self.beta2_input.setDecimals(4)
        self.beta2_input.setRange(0.0001, 0.9999)
        self.beta2_input.setValue(0.5000)
        self.main_layout.addWidget(self.beta2_label, 7, 0)
        self.main_layout.addWidget(self.beta2_input, 7, 1)
        self.constraint_level_label = QtWidgets.QLabel(self.tr("Constraint Level"))
        self.constraint_level_label.setToolTip(self.tr(
            "It controls the constraint intensity of the end member diversity between different samples. "
            "When the constraint level is high, the end members of different samples tend to be equal."))
        self.constraint_level_input = QtWidgets.QDoubleSpinBox()
        self.constraint_level_input.setDecimals(4)
        self.constraint_level_input.setRange(-10, 10.0)
        self.constraint_level_input.setValue(2.0)
        self.main_layout.addWidget(self.constraint_level_label, 8, 0)
        self.main_layout.addWidget(self.constraint_level_input, 8, 1)
        self.need_history_checkbox = QtWidgets.QCheckBox(self.tr("Need History"))
        self.need_history_checkbox.setToolTip(self.tr(
            "If checked, it will record the variation history of model."))
        self.need_history_checkbox.setChecked(False)
        self.main_layout.addWidget(self.need_history_checkbox, 9, 0, 1, 2)
        self._update_device_list()

    @property
    def settings(self) -> Dict[str, Any]:
        settings = dict(
            device=self.device_combo_box.currentText(),
            pretrain_epochs=self.pretrain_epochs_input.value(),
            min_epochs=self.min_epochs_input.value(),
            max_epochs=self.max_epochs_input.value(),
            precision=self.precision_input.value(),
            learning_rate=self.learning_rate_input.value() / 1000.0,
            betas=(self.beta1_input.value(), self.beta2_input.value()),
            constraint_level=self.constraint_level_input.value(),
            need_history=self.need_history_checkbox.isChecked())
        return settings

    @settings.setter
    def settings(self, s: Dict):
        self.device_combo_box.setCurrentIndex(s["device"])
        self.pretrain_epochs_input.setValue(s["pretrain_epochs"])
        self.min_epochs_input.setValue(s["min_epochs"])
        self.max_epochs_input.setValue(s["max_epochs"])
        self.precision_input.setValue(s["precision"])
        self.learning_rate_input.setValue(s["learning_rate"] * 1000.0)
        beta1, beta2 = s["betas"]
        self.beta1_input.setValue(beta1)
        self.beta2_input.setValue(beta2)
        self.constraint_level_input.setValue(s["constraint_level"])
        self.need_history_checkbox.setChecked(s["need_history"])

    def _update_device_list(self):
        if self._client is not None:
            try:
                server_state = self._client.get_service_state()
                available_devices = server_state["available_devices"]
                self.device_combo_box.addItems(available_devices)
                return
            except _InactiveRpcError:
                self.logger.warning("The remote grpc server is not available.")

        import torch
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
        available_devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        self.device_combo_box.addItems(available_devices)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("UDM Settings"))
        self.device_label.setText(self.tr("Device"))
        self.device_label.setToolTip(self.tr(
            "The neural network framework, pytorch, also can use the GPU of NVIDIA to do calculations."))
        self.pretrain_epochs_label.setText(self.tr("Pretrain Epochs"))
        self.pretrain_epochs_label.setToolTip(self.tr(
            "The number of epochs before formal training. "
            "The pretrain process only update the proportions of end members."))
        self.min_epochs_label.setText(self.tr("Minimum Number of Epochs"))
        self.min_epochs_label.setToolTip(self.tr("Minimum number of epochs to be performed."))
        self.max_epochs_label.setText(self.tr("Maximum Number of Epochs"))
        self.max_epochs_label.setToolTip(self.tr("Maximum number of epochs to be performed."))
        self.precision_label.setText(self.tr("Precision"))
        self.precision_label.setToolTip(self.tr(
            "It controls the precision for the value of loss function in the stopping criterion."))
        self.learning_rate_label.setText(self.tr("Learning Rate (x10<sup>-3</sup>)"))
        self.learning_rate_label.setToolTip(self.tr(
            "The learning rate of the neural network to update its weights from gradient."))
        self.beta1_label.setText(self.tr("Beta 1"))
        self.beta1_label.setToolTip(self.tr(
            "Betas are the coefficients used for computing running averages of gradient and its square."))
        self.beta2_label.setText(self.tr("Beta 2"))
        self.beta2_label.setToolTip(self.tr(
            "Betas are the coefficients used for computing running averages of gradient and its square."))
        self.constraint_level_label.setText(self.tr("Constraint Level"))
        self.constraint_level_label.setToolTip(
            "It controls the constraint intensity of the end member diversity between different samples. "
            "When the constraint level is high, the end members of different samples tend to be equal.")
        self.need_history_checkbox.setText(self.tr("Need History"))
        self.need_history_checkbox.setToolTip(self.tr(
            "If checked, it will record the variation history of model."))
