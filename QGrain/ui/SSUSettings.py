__all__ = ["SSUSettings"]

from typing import *

from PySide6 import QtCore, QtWidgets


class SSUSettings(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("SSU Settings"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.loss_label = QtWidgets.QLabel(self.tr("Loss Function"))
        self.loss_label.setToolTip(self.tr(
            "The function to calculate the difference between prediction and observation."))
        self.loss_combo_box = QtWidgets.QComboBox()
        for key, name in self.supported_losses:
            self.loss_combo_box.addItem(name)
        self.loss_combo_box.setCurrentIndex(8)
        self.main_layout.addWidget(self.loss_label, 0, 0)
        self.main_layout.addWidget(self.loss_combo_box, 0, 1)
        self.optimizer_label = QtWidgets.QLabel(self.tr("Optimizer"))
        self.optimizer_label.setToolTip(self.tr("The optimizer to find the minimum of loss function."))
        self.optimizer_combo_box = QtWidgets.QComboBox()
        for key, name in self.supported_optimizers:
            self.optimizer_combo_box.addItem(name)
        self.optimizer_combo_box.setCurrentIndex(6)
        self.main_layout.addWidget(self.optimizer_label, 1, 0)
        self.main_layout.addWidget(self.optimizer_combo_box, 1, 1)
        self.try_global_checkbox = QtWidgets.QCheckBox(self.tr("Global Optimization"))
        self.try_global_checkbox.setChecked(False)
        self.try_global_checkbox.setToolTip(self.tr("Try global optimization or not."))
        self.try_global_checkbox.clicked.connect(self.on_try_global_changed)
        self.main_layout.addWidget(self.try_global_checkbox, 2, 0, 1, 2)
        self.global_max_niter_label = QtWidgets.QLabel(self.tr("Maximum Number of Iterations"))
        self.global_max_niter_label.setToolTip(self.tr("Maximum number of iterations of global optimization."))
        self.global_max_niter_input = QtWidgets.QSpinBox()
        self.global_max_niter_input.setRange(0, 10000)
        self.global_max_niter_input.setValue(100)
        self.main_layout.addWidget(self.global_max_niter_label, 3, 0)
        self.main_layout.addWidget(self.global_max_niter_input, 3, 1)
        self.global_niter_success_label = QtWidgets.QLabel(self.tr("Success Number of Iterations"))
        self.global_niter_success_label.setToolTip(self.tr(
            "The number of iterations that reaching the same minimum value."))
        self.global_niter_success_input = QtWidgets.QSpinBox()
        self.global_niter_success_input.setRange(1, 100)
        self.global_niter_success_input.setValue(5)
        self.main_layout.addWidget(self.global_niter_success_label, 4, 0)
        self.main_layout.addWidget(self.global_niter_success_input, 4, 1)
        self.global_step_size_label = QtWidgets.QLabel(self.tr("Step Size"))
        self.global_step_size_label.setToolTip(self.tr("The step size of searching the global minimum."))
        self.global_step_size_input = QtWidgets.QDoubleSpinBox()
        self.global_step_size_input.setRange(0.01, 10)
        self.global_step_size_input.setValue(0.1)
        self.main_layout.addWidget(self.global_step_size_label, 5, 0)
        self.main_layout.addWidget(self.global_step_size_input, 5, 1)
        self.optimizer_max_niter_label = QtWidgets.QLabel(self.tr("Maximum Number of Iterations of Optimizer"))
        self.optimizer_max_niter_label.setToolTip(self.tr(
            "Maximum number of iterations of the optimizer in global and local optimization."))
        self.optimizer_max_niter_input = QtWidgets.QSpinBox()
        self.optimizer_max_niter_input.setRange(0, 100000)
        self.optimizer_max_niter_input.setValue(500)
        self.main_layout.addWidget(self.optimizer_max_niter_label, 6, 0)
        self.main_layout.addWidget(self.optimizer_max_niter_input, 6, 1)
        self.need_history_checkbox = QtWidgets.QCheckBox(self.tr("Need History"))
        self.need_history_checkbox.setChecked(True)
        self.need_history_checkbox.setToolTip(self.tr("Record the variation history of parameters or not."))
        self.main_layout.addWidget(self.need_history_checkbox, 7, 0, 1, 2)

    @property
    def supported_losses(self) -> Sequence[Tuple[str, str]]:
        losses = (("1-norm", self.tr("1 Norm")),
                  ("2-norm", self.tr("2 Norm")),
                  ("3-norm", self.tr("3 Norm")),
                  ("4-norm", self.tr("4 Norm")),
                  ("mae", self.tr("MAE")),
                  ("mse", self.tr("MSE")),
                  ("rmse", self.tr("RMSE")),
                  ("rmlse", self.tr("RMLSE")),
                  ("lmse", self.tr("LMSE")),
                  ("cosine", self.tr("Cosine")),
                  ("angular", self.tr("Angular")))
        return losses

    @property
    def supported_optimizers(self) -> Sequence[Tuple[str, str]]:
        optimizers = (("Nelder-Mead", self.tr("Nelder-Mead")),
                      ("Powell", self.tr("Powell")),
                      ("CG", self.tr("CG")),
                      ("BFGS", self.tr("BFGS")),
                      ("L-BFGS-B", self.tr("L-BFGS-B")),
                      ("TNC", self.tr("TNC")),
                      ("SLSQP", self.tr("SLSQP")))
        return optimizers

    @property
    def loss(self) -> str:
        return self.supported_losses[self.loss_combo_box.currentIndex()][0]

    @property
    def optimizer(self) -> str:
        return self.supported_optimizers[self.optimizer_combo_box.currentIndex()][0]

    @property
    def settings(self) -> Dict:
        s = dict(
            loss=self.loss,
            optimizer=self.optimizer,
            try_global=self.try_global_checkbox.isChecked(),
            global_max_niter=self.global_max_niter_input.value(),
            global_niter_success=self.global_niter_success_input.value(),
            global_step_size=self.global_step_size_input.value(),
            optimizer_max_niter=self.optimizer_max_niter_input.value(),
            need_history=self.need_history_checkbox.isChecked())
        return s

    @settings.setter
    def settings(self, s: Dict[str, Any]):
        loss_map = {key: i for i, (key, name) in enumerate(self.supported_losses)}
        assert s["loss"] in loss_map
        optimizer_map = {key: i for i, (key, name) in enumerate(self.supported_optimizers)}
        assert s["optimizer"] in optimizer_map
        self.loss_combo_box.setCurrentIndex(loss_map[s["loss"]])
        self.optimizer_combo_box.setCurrentIndex(optimizer_map[s["optimizer"]])
        self.try_global_checkbox.setChecked(s["try_global"])
        self.global_max_niter_input.setValue(s["global_max_niter"])
        self.global_niter_success_input.setValue(s["global_niter_success"])
        self.global_step_size_input.setValue(s["global_step_size"])
        self.optimizer_max_niter_input.setValue(s["optimizer_max_niter"])
        self.need_history_checkbox.setChecked(s["need_history"])

    def on_try_global_changed(self):
        checked = self.try_global_checkbox.isChecked()
        self.global_max_niter_input.setEnabled(checked)
        self.global_niter_success_input.setEnabled(checked)
        self.global_step_size_input.setEnabled(checked)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("SSU Settings"))
        self.loss_label.setText(self.tr("Loss Function"))
        self.loss_label.setToolTip(self.tr(
            "The function to calculate the difference between prediction and observation."))
        for i, (key, name) in enumerate(self.supported_losses):
            self.loss_combo_box.setItemText(i, name)
        self.optimizer_label.setText(self.tr("Optimizer"))
        self.optimizer_label.setToolTip(self.tr("The optimizer to find the minimum of loss function."))
        for i, (key, name) in enumerate(self.supported_optimizers):
            self.optimizer_combo_box.setItemText(i, name)
        self.try_global_checkbox.setText(self.tr("Global Optimization"))
        self.try_global_checkbox.setToolTip(self.tr("Try global optimization or not."))
        self.global_max_niter_label.setText(self.tr("Maximum Number of Iterations"))
        self.global_max_niter_label.setToolTip(self.tr("Maximum number of iterations of global optimization."))
        self.global_niter_success_label.setText(self.tr("Success Number of Iterations"))
        self.global_niter_success_label.setToolTip(self.tr(
            "The number of iterations that reaching the same minimum value."))
        self.global_step_size_label.setText(self.tr("Step Size"))
        self.global_step_size_label.setToolTip(self.tr("The step size of searching the global minimum."))
        self.optimizer_max_niter_label.setText(self.tr("Maximum Number of Iterations of Optimizer"))
        self.optimizer_max_niter_label.setToolTip(self.tr(
            "Maximum number of iterations of the optimizer in global and local optimization."))
        self.need_history_checkbox.setText(self.tr("Need History"))
        self.need_history_checkbox.setToolTip(self.tr("Record the variation history of parameters or not."))
