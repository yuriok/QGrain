__all__ = ["SSUAnalyzer"]

from typing import *

from PySide6 import QtCore, QtWidgets

from ..models import DistributionType, Dataset, SSUResult
from ..charts.DistributionChart import DistributionChart
from .ParameterEditor import ParameterEditor
from .SSUResultViewer import SSUResultViewer
from .SSUSettings import SSUSettings
from ..ssu import try_ssu


class SSUAnalyzer(QtWidgets.QWidget):
    SUPPORT_DISTRIBUTIONS = (
        (DistributionType.Normal, "Normal"),
        (DistributionType.SkewNormal, "Skew Normal"),
        (DistributionType.Weibull, "Weibull"),
        (DistributionType.GeneralWeibull, "General Weibull"))

    def __init__(self, setting_dialog: SSUSettings, parameter_editor: ParameterEditor, parent=None):
        super().__init__(parent=parent)
        assert setting_dialog is not None
        assert parameter_editor is not None
        self.setting_dialog = setting_dialog
        self.parameter_editor = parameter_editor
        self.setWindowTitle(self.tr("SSU Analyzer"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.control_group = QtWidgets.QGroupBox(self.tr("Control"))
        self.control_layout = QtWidgets.QGridLayout(self.control_group)
        self.distribution_label = QtWidgets.QLabel(self.tr("Distribution Type"))
        self.distribution_combo_box = QtWidgets.QComboBox()
        self.distribution_combo_box.addItems([name for _, name in self.SUPPORT_DISTRIBUTIONS])
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
        self.chart_group = QtWidgets.QGroupBox(self.tr("Chart"))
        self.chart_layout = QtWidgets.QGridLayout(self.chart_group)
        self.result_chart = DistributionChart()
        self.chart_layout.addWidget(self.result_chart, 0, 0)
        self.result_group = QtWidgets.QGroupBox(self.tr("Result"))
        self.result_layout = QtWidgets.QGridLayout(self.result_group)
        self.result_view = SSUResultViewer()
        self.result_view.result_displayed.connect(self.on_result_displayed)
        self.result_view.result_referred.connect(self.parameter_editor.refer_ssu_result)
        self.result_layout.addWidget(self.result_view)
        self.splitter_1 = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter_1.addWidget(self.control_group)
        self.splitter_1.addWidget(self.chart_group)
        self.splitter_2 = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter_2.addWidget(self.splitter_1)
        self.splitter_2.addWidget(self.result_group)
        self.splitter_2.setStretchFactor(0, 1)
        self.splitter_2.setStretchFactor(1, 2)
        self.main_layout.addWidget(self.splitter_2, 0, 0)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self._dataset = None

    @property
    def distribution_type(self) -> DistributionType:
        distribution_type = self.SUPPORT_DISTRIBUTIONS[self.distribution_combo_box.currentIndex()][0]
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

    def on_dataset_loaded(self, dataset: Dataset):
        self._dataset = dataset
        self.sample_index_input.setRange(1, len(dataset))
        self.sample_index_input.setValue(1)
        self.on_sample_index_changed(1)
        self.sample_index_input.setEnabled(True)
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)
        self.try_previous_button.setEnabled(True)
        self.try_next_button.setEnabled(True)

    def on_sample_index_changed(self, index):
        self.sample_name_display.setText(self._dataset[index - 1].name)

    def get_task(self, sample_index: int) -> Dict[str, Any]:
        task = {}
        sample = self._dataset[sample_index]
        task["sample"] = sample
        if self.parameter_editor.parameter_enabled:
            task["distribution_type"] = self.parameter_editor.distribution_type
            task["n_components"] = self.parameter_editor.n_components
            task["x0"] = self.parameter_editor.parameters
        else:
            task["distribution_type"] = self.distribution_type
            task["n_components"] = self.n_components
        task.update(self.setting_dialog.settings)
        return task

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        for i in range(len(self._dataset)):
            task = self.get_task(i)
            tasks.append(task)
        return tasks

    def on_result_displayed(self, result: SSUResult):
        self.result_chart.show_result(result)

    def do_test(self):
        self.try_fit_button.setEnabled(False)
        self.edit_parameter_button.setEnabled(False)
        self.try_previous_button.setEnabled(False)
        self.try_next_button.setEnabled(False)
        task = self.get_task(self.sample_index_input.value()-1)
        result, msg = try_ssu(**task)
        if isinstance(result, SSUResult):
            self.result_chart.show_result(result)
            self.result_view.add_result(result)
        else:
            self.show_error(self.tr("The SSU fitting task failed: {0}.").format(msg))
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)
        self.try_previous_button.setEnabled(True)
        self.try_next_button.setEnabled(True)

    def on_edit_parameter_clicked(self):
        if self._dataset is not None:
            sample_index = self.sample_index_input.value()-1
            sample = self._dataset[sample_index]
            self.parameter_editor.setup_target(sample.classes, sample.distribution)
        self.parameter_editor.show()
        self.parameter_editor.update_chart()

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
        if current == len(self._dataset):
            return
        self.sample_index_input.setValue(current+1)
        self.do_test()

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("SSU Analyzer"))
        self.control_group.setTitle(self.tr("Control"))
        self.distribution_label.setText(self.tr("Distribution Type"))
        self.component_number_label.setText(self.tr("Number of Components"))
        self.sample_index_label.setText(self.tr("Sample Index"))
        self.sample_name_label.setText(self.tr("Sample Name"))
        if self._dataset is None:
            self.sample_name_display.setText(self.tr("Unknown"))
        self.try_fit_button.setText(self.tr("Try Fit"))
        self.edit_parameter_button.setText(self.tr("Edit Parameters"))
        self.try_previous_button.setText(self.tr("Try Previous"))
        self.try_next_button.setText(self.tr("Try Next"))
        self.chart_group.setTitle(self.tr("Chart"))
        self.result_group.setTitle(self.tr("Result"))
        self.result_view.retranslate()
