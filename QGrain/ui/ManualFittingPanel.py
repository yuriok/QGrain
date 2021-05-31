__all__ = ["ManualFittingPanel"]

import copy

import numpy as np
import qtawesome as qta
from PySide2.QtCore import Qt, QTimer, Signal
from PySide2.QtWidgets import (QDialog, QDoubleSpinBox, QGridLayout, QGroupBox,
                               QLabel, QMessageBox, QPushButton, QSlider,
                               QSplitter)
from QGrain import DistributionType
from QGrain.charts.MixedDistributionChart import MixedDistributionChart
from QGrain.distributions import (BaseDistribution, NormalDistribution,
                                  SkewNormalDistribution, WeibullDistribution)
from QGrain.ssu import AsyncWorker, SSUResult, SSUTask


class ManualFittingPanel(QDialog):
    manual_fitting_finished = Signal(SSUResult)
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("SSU Manual Fitting Panel"))
        self.control_widgets = []
        self.input_widgets = []
        self.last_task = None
        self.last_result = None
        self.async_worker = AsyncWorker()
        self.async_worker.background_worker.task_succeeded.connect(self.on_task_succeeded)
        self.initialize_ui()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.update_chart)
        self.chart_timer.setSingleShot(True)

    def initialize_ui(self):
        self.main_layout = QGridLayout(self)

        self.chart_group = QGroupBox(self.tr("Chart"))
        self.chart_layout = QGridLayout(self.chart_group)
        self.chart = MixedDistributionChart(show_mode=True, toolbar=False)
        self.chart_layout.addWidget(self.chart)

        self.control_group = QGroupBox(self.tr("Control"))
        self.control_layout = QGridLayout(self.control_group)
        self.try_button = QPushButton(qta.icon("mdi.test-tube"), self.tr("Try"))
        self.try_button.clicked.connect(self.on_try_clicked)
        self.control_layout.addWidget(self.try_button, 1, 0, 1, 4)
        self.confirm_button = QPushButton(qta.icon("ei.ok-circle"), self.tr("Confirm"))
        self.confirm_button.clicked.connect(self.on_confirm_clicked)
        self.control_layout.addWidget(self.confirm_button, 2, 0, 1, 4)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.chart_group)
        self.splitter.addWidget(self.control_group)
        self.main_layout.addWidget(self.splitter)

    def change_n_components(self, n_components: int):
        for widget in self.control_widgets:
            self.control_layout.removeWidget(widget)
            widget.hide()
        self.control_widgets.clear()
        self.input_widgets.clear()

        widgets = []
        slider_range = (0, 1000)
        input_widgets = []
        mean_range = (-5, 15)
        std_range = (0.0, 10)
        weight_range = (0, 10)
        names = [self.tr("Mean"), self.tr("STD"), self.tr("Weight")]
        ranges = [mean_range, std_range, weight_range]
        slider_values = [500, 100, 100]
        input_values = [0.0, 1.0, 1.0]

        for i in range(n_components):
            group = QGroupBox(f"C{i+1}")
            group.setMinimumWidth(200)
            group_layout = QGridLayout(group)
            inputs = []
            for j, (name, range_, slider_value, input_value) in enumerate(zip(names, ranges, slider_values, input_values)):
                label = QLabel(name)
                slider = QSlider()
                slider.setRange(*slider_range)
                slider.setValue(slider_value)
                slider.setOrientation(Qt.Horizontal)
                input_ = QDoubleSpinBox()
                input_.setRange(*range_)
                input_.setDecimals(3)
                input_.setSingleStep(0.01)
                input_.setValue(input_value)
                slider.valueChanged.connect(self.on_value_changed)
                input_.valueChanged.connect(self.on_value_changed)
                slider.valueChanged.connect(lambda x, input_=input_, range_=range_: input_.setValue(x/1000*(range_[-1]-range_[0])+range_[0]))
                input_.valueChanged.connect(lambda x, slider=slider, range_=range_: slider.setValue((x-range_[0])/(range_[-1]-range_[0])*1000))

                group_layout.addWidget(label, j, 0)
                group_layout.addWidget(slider, j, 1)
                group_layout.addWidget(input_, j, 2)
                inputs.append(input_)

            self.control_layout.addWidget(group, i+5, 0, 1, 4)
            widgets.append(group)
            input_widgets.append(inputs)

        self.control_widgets = widgets
        self.input_widgets = input_widgets

    @property
    def n_components(self) -> int:
        return len(self.input_widgets)

    @property
    def expected(self):
        reference = []
        weights = []
        for i, (mean, std, weight) in enumerate(self.input_widgets):
            reference.append(dict(mean=mean.value(), std=std.value(), skewness=0.0))
            weights.append(weight.value())
        weights = np.array(weights)
        fractions = weights / np.sum(weights)
        return reference, fractions

    def show_message(self, title: str, message: str):
        self.msg_box.setWindowTitle(title)
        self.msg_box.setText(message)
        self.msg_box.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    def on_confirm_clicked(self):
        if self.last_result is not None:
            for component, (mean, std, weight) in zip(self.last_result.components, self.input_widgets):
                mean.setValue(component.logarithmic_moments["mean"])
                std.setValue(component.logarithmic_moments["std"])
                weight.setValue(component.fraction*10)
            self.manual_fitting_finished.emit(self.last_result)

            self.last_result = None
            self.last_task = None
            self.try_button.setEnabled(False)
            self.confirm_button.setEnabled(False)
            self.hide()

    def on_task_failed(self, info: str, task: SSUTask):
        self.show_error(info)

    def on_task_succeeded(self, result: SSUResult):
        self.chart.show_model(result.view_model)
        self.last_result = result
        self.confirm_button.setEnabled(True)

    def on_try_clicked(self):
        if self.last_task is None:
            return
        new_task = copy.copy(self.last_task)
        reference, fractions = self.expected
        initial_guess = BaseDistribution.get_initial_guess(self.last_task.distribution_type, reference, fractions=fractions)
        new_task.initial_guess = initial_guess
        self.async_worker.execute_task(new_task)

    def on_value_changed(self):
        self.chart_timer.stop()
        self.chart_timer.start(10)

    def update_chart(self):
        if self.last_task is None:
            return
        reference, fractions = self.expected
        for comp_ref in reference:
            if comp_ref["std"] == 0.0:
                return
        # print(reference)
        initial_guess = BaseDistribution.get_initial_guess(self.last_task.distribution_type, reference, fractions=fractions)
        result = SSUResult(self.last_task, initial_guess)
        self.chart.show_model(result.view_model, quick=True)

    def setup_task(self, task: SSUTask):
        self.last_task = task
        self.try_button.setEnabled(True)
        if self.n_components != task.n_components:
            self.change_n_components(task.n_components)
        reference, fractions = self.expected
        initial_guess = BaseDistribution.get_initial_guess(task.distribution_type, reference, fractions=fractions)
        result = SSUResult(task, initial_guess)
        self.chart.show_model(result.view_model, quick=False)


if __name__ == "__main__":
    import sys

    from QGrain.artificial import get_random_sample
    from QGrain.entry import setup_app

    app, splash = setup_app()
    main = ManualFittingPanel()
    main.show()
    splash.finish(main)
    sample = get_random_sample().sample_to_fit
    task = SSUTask(sample, DistributionType.Normal, 3)
    main.setup_task(task)
    sys.exit(app.exec_())
