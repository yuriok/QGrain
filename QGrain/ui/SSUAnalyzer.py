import typing

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..chart.DistributionChart import DistributionChart
from ..model import GrainSizeDataset
from ..ssu import AsyncWorker, DistributionType, SSUResult, SSUTask
from ..ui.FittingResultViewer import FittingResultViewer
from ..ui.ReferenceResultViewer import ReferenceResultViewer
from .ParameterEditor import ParameterEditor
from .SSUSettingDialog import SSUSettingDialog


class SSUAnalyzer(QtWidgets.QWidget):
    SUPPORT_DISTRIBUTIONS = (
        DistributionType.Normal,
        DistributionType.SkewNormal,
        DistributionType.Weibull,
        DistributionType.GeneralWeibull)
    def __init__(self, setting_dialog: SSUSettingDialog, parameter_editor: ParameterEditor, parent=None):
        super().__init__(parent=parent)
        assert setting_dialog is not None
        assert parameter_editor is not None
        self.setting_dialog = setting_dialog
        self.parameter_editor = parameter_editor
        self.async_worker = AsyncWorker()
        self.async_worker.background_worker.task_succeeded.connect(self.on_fitting_succeeded)
        self.async_worker.background_worker.task_failed.connect(self.on_fitting_failed)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.dataset = None
        self.task_table = {}
        self.task_results = {}
        self.failed_task_ids = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("SSU Analyzer"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        # control group
        self.control_group = QtWidgets.QGroupBox(self.tr("Control"))
        self.control_layout = QtWidgets.QGridLayout(self.control_group)
        self.distribution_label = QtWidgets.QLabel(self.tr("Distribution Type"))
        self.distribution_combo_box = QtWidgets.QComboBox()
        self.distribution_combo_box.addItems([distribution_type.value for distribution_type in self.SUPPORT_DISTRIBUTIONS])
        self.component_number_label = QtWidgets.QLabel(self.tr("Number of Components"))
        self.n_components_input = QtWidgets.QSpinBox()
        self.n_components_input.setRange(1, 10)
        self.n_components_input.setValue(3)
        self.control_layout.addWidget(self.distribution_label, 0, 0)
        self.control_layout.addWidget(self.distribution_combo_box, 0, 1)
        self.control_layout.addWidget(self.component_number_label, 1, 0)
        self.control_layout.addWidget(self.n_components_input, 1, 1)

        self.sample_index_label = QtWidgets.QLabel(self.tr("Sample Index"))
        self.sample_index_input = QtWidgets.QSpinBox()
        self.sample_index_input.valueChanged.connect(self.on_sample_index_changed)
        self.sample_index_input.setEnabled(False)
        self.control_layout.addWidget(self.sample_index_label, 2, 0)
        self.control_layout.addWidget(self.sample_index_input, 2, 1)
        self.sample_name_label = QtWidgets.QLabel(self.tr("Sample Name"))
        self.sample_name_display = QtWidgets.QLabel(self.tr("Unknown"))
        self.control_layout.addWidget(self.sample_name_label, 3, 0)
        self.control_layout.addWidget(self.sample_name_display, 3, 1)

        self.try_fit_button = QtWidgets.QPushButton(self.tr("Try Fit"))
        self.try_fit_button.setEnabled(False)
        self.try_fit_button.clicked.connect(self.on_try_fit_clicked)
        self.edit_parameter_button = QtWidgets.QPushButton(self.tr("Edit Parameters"))
        self.edit_parameter_button.clicked.connect(self.on_edit_parameter_clicked)
        self.control_layout.addWidget(self.try_fit_button, 4, 0)
        self.control_layout.addWidget(self.edit_parameter_button, 4, 1)

        self.try_previous_button = QtWidgets.QPushButton(self.tr("Try Previous"))
        self.try_previous_button.setEnabled(False)
        self.try_previous_button.clicked.connect(self.on_try_previous_clicked)
        self.try_next_button = QtWidgets.QPushButton(self.tr("Try Next"))
        self.try_next_button.setEnabled(False)
        self.try_next_button.clicked.connect(self.on_try_next_clicked)
        self.control_layout.addWidget(self.try_previous_button, 5, 0)
        self.control_layout.addWidget(self.try_next_button, 5, 1)

        # chart group
        self.chart_group = QtWidgets.QGroupBox(self.tr("Chart"))
        self.chart_layout = QtWidgets.QGridLayout(self.chart_group)
        self.result_chart = DistributionChart(show_mode=True)
        self.chart_layout.addWidget(self.result_chart, 0, 0)

        # table group
        self.reference_view = ReferenceResultViewer()
        self.result_view = FittingResultViewer(self.reference_view)
        self.result_view.result_marked.connect(lambda result: self.reference_view.add_references([result]))
        self.table_tab = QtWidgets.QTabWidget()
        self.table_tab.addTab(self.result_view, self.tr("Result"))
        self.table_tab.addTab(self.reference_view, self.tr("Reference"))

        # pack all group
        self.splitter_1 = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter_1.addWidget(self.control_group)
        self.splitter_1.addWidget(self.chart_group)
        self.splitter_2 = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter_2.addWidget(self.splitter_1)
        self.splitter_2.addWidget(self.table_tab)
        self.splitter_2.setStretchFactor(0, 1)
        self.splitter_2.setStretchFactor(1, 2)
        self.main_layout.addWidget(self.splitter_2, 0, 0)

    @property
    def distribution_type(self) -> DistributionType:
        distribution_type = self.SUPPORT_DISTRIBUTIONS[self.distribution_combo_box.currentIndex()]
        return distribution_type

    @property
    def n_components(self) -> int:
        return self.n_components_input.value()

    def show_message(self, title: str, message: str):
        self.normal_msg.setWindowTitle(title)
        self.normal_msg.setText(message)
        self.normal_msg.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    def on_dataset_loaded(self, dataset: GrainSizeDataset):
        self.dataset = dataset
        self.sample_index_input.setRange(1, dataset.n_samples)
        self.sample_index_input.setEnabled(True)
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)
        self.try_previous_button.setEnabled(True)
        self.try_next_button.setEnabled(True)

    def on_sample_index_changed(self, index):
        self.sample_name_display.setText(self.dataset.samples[index-1].name)

    def generate_task(self):
        sample_index = self.sample_index_input.value()-1
        sample = self.dataset.samples[sample_index]
        setting = self.setting_dialog.setting
        if self.parameter_editor.parameter_enabled:
            task = SSUTask(
                sample,
                self.parameter_editor.distribution_type,
                self.parameter_editor.n_components,
                resolver_setting=setting,
                initial_guess=self.parameter_editor.parameters
            )
        else:
            query = self.reference_view.query_reference(sample) # type: SSUResult
            if query is not None:
                task = SSUTask(
                    sample,
                    query.distribution_type,
                    query.n_components,
                    resolver_setting=setting,
                    initial_guess=query.func_args)
            else:
                task = SSUTask(
                    sample,
                    self.distribution_type,
                    self.n_components,
                    resolver_setting=setting)
        return task

    def on_fitting_succeeded(self, fitting_result: SSUResult):
        # update chart
        self.result_chart.show_model(fitting_result.view_model)
        self.result_view.add_result(fitting_result)
        self.task_results[fitting_result.task.uuid] = fitting_result
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)
        self.try_previous_button.setEnabled(True)
        self.try_next_button.setEnabled(True)

    def on_fitting_failed(self, failed_info: str, task: SSUTask):
        self.failed_task_ids.append(task.uuid)
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)
        self.try_previous_button.setEnabled(True)
        self.try_next_button.setEnabled(True)
        self.show_error(failed_info)

    def do_test(self):
        self.try_fit_button.setEnabled(False)
        self.edit_parameter_button.setEnabled(False)
        self.try_previous_button.setEnabled(False)
        self.try_next_button.setEnabled(False)
        task = self.generate_task()
        self.task_table[task.uuid] = task
        self.async_worker.execute_task(task)

    def on_edit_parameter_clicked(self):
        if self.dataset is not None:
            sample_index = self.sample_index_input.value()-1
            sample = self.dataset.samples[sample_index]
            self.parameter_editor.setup_target(sample.classes_μm, sample.distribution)
        self.parameter_editor.show()

    def on_try_fit_clicked(self):
        self.do_test()

    def on_try_previous_clicked(self):
        current = self.sample_index_input.value()
        if current == 1:
            return
        self.sample_index_input.setValue(current-1)
        self.do_test()

    def on_try_next_clicked(self):
        current = self.sample_index_input.value()
        if current == self.dataset.n_samples:
            return
        self.sample_index_input.setValue(current+1)
        self.do_test()

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("SSU Analyzer"))
