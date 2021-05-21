import typing

import numpy as np
import qtawesome as qta
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QComboBox, QDialog, QGridLayout, QGroupBox, QTabWidget,
                               QLabel, QMessageBox, QPushButton, QSpinBox,
                               QSplitter)
from QGrain.algorithms import DistributionType
from QGrain.algorithms.AsyncFittingWorker import AsyncFittingWorker
from QGrain.algorithms.moments import logarithmic
from QGrain.charts.MixedDistributionChart import MixedDistributionChart
from QGrain.models.FittingResult import FittingResult
from QGrain.models.FittingTask import FittingTask
from QGrain.models.GrainSizeDataset import GrainSizeDataset
from QGrain.ui.ClassicResolverSettingWidget import ClassicResolverSettingWidget
from QGrain.ui.FittingResultViewer import FittingResultViewer
from QGrain.ui.LoadDatasetDialog import LoadDatasetDialog
from QGrain.ui.ManualFittingPanel import ManualFittingPanel
from QGrain.ui.NNResolverSettingWidget import NNResolverSettingWidget
from QGrain.ui.ReferenceResultViewer import ReferenceResultViewer


class SSUResolverPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("SSU Resolver"))
        self.distribution_types = [(DistributionType.Normal, self.tr("Normal")),
                                   (DistributionType.Weibull, self.tr("Weibull")),
                                   (DistributionType.SkewNormal, self.tr("Skew Normal"))]
        self.load_dataset_dialog = LoadDatasetDialog(parent=self)
        self.load_dataset_dialog.dataset_loaded.connect(self.on_dataset_loaded)
        self.classic_setting = ClassicResolverSettingWidget(parent=self)
        self.neural_setting = NNResolverSettingWidget(parent=self)
        self.manual_panel = ManualFittingPanel(parent=self)
        self.manual_panel.manual_fitting_finished.connect(self.on_fitting_succeeded)
        self.async_worker = AsyncFittingWorker()
        self.async_worker.background_worker.task_succeeded.connect(self.on_fitting_succeeded)
        self.async_worker.background_worker.task_failed.connect(self.on_fitting_failed)
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.dataset = None
        self.task_table = {}
        self.task_results = {}
        self.failed_task_ids = []
        self.__continuous_flag = False
        self.init_ui()

    def init_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.main_layout = QGridLayout(self)
        # self.main_layout.setContentsMargins(0, 0, 0, 0)
        # control group
        self.control_group = QGroupBox(self.tr("Control"))
        self.control_layout = QGridLayout(self.control_group)
        self.resolver_label = QLabel(self.tr("Resolver"))
        self.resolver_combo_box = QComboBox()
        self.resolver_combo_box.addItems(["classic", "neural"])
        self.control_layout.addWidget(self.resolver_label, 0, 0)
        self.control_layout.addWidget(self.resolver_combo_box, 0, 1)
        self.load_dataset_button = QPushButton(qta.icon("fa.database"), self.tr("Load Dataset"))
        self.load_dataset_button.clicked.connect(self.on_load_dataset_clicked)
        self.configure_fitting_button = QPushButton(qta.icon("fa.gears"), self.tr("Configure Fitting Algorithm"))
        self.configure_fitting_button.clicked.connect(self.on_configure_fitting_clicked)
        self.control_layout.addWidget(self.load_dataset_button, 1, 0)
        self.control_layout.addWidget(self.configure_fitting_button, 1, 1)
        self.distribution_label = QLabel(self.tr("Distribution Type"))
        self.distribution_combo_box = QComboBox()
        self.distribution_combo_box.addItems([name for _, name in self.distribution_types])
        self.component_number_label = QLabel(self.tr("N<sub>components</sub>"))
        self.n_components_input = QSpinBox()
        self.n_components_input.setRange(1, 10)
        self.n_components_input.setValue(3)
        self.control_layout.addWidget(self.distribution_label, 2, 0)
        self.control_layout.addWidget(self.distribution_combo_box, 2, 1)
        self.control_layout.addWidget(self.component_number_label, 3, 0)
        self.control_layout.addWidget(self.n_components_input, 3, 1)

        self.n_samples_label = QLabel(self.tr("N<sub>samples</sub>"))
        self.n_samples_display = QLabel(self.tr("Unknown"))
        self.control_layout.addWidget(self.n_samples_label, 4, 0)
        self.control_layout.addWidget(self.n_samples_display, 4, 1)
        self.sample_index_label = QLabel(self.tr("Sample Index"))
        self.sample_index_input = QSpinBox()
        self.sample_index_input.valueChanged.connect(self.on_sample_index_changed)
        self.sample_index_input.setEnabled(False)
        self.control_layout.addWidget(self.sample_index_label, 5, 0)
        self.control_layout.addWidget(self.sample_index_input, 5, 1)
        self.sample_name_label = QLabel(self.tr("Sample Name"))
        self.sample_name_display = QLabel(self.tr("Unknown"))
        self.control_layout.addWidget(self.sample_name_label, 6, 0)
        self.control_layout.addWidget(self.sample_name_display, 6, 1)

        self.manual_test_button = QPushButton(qta.icon("fa.sliders"), self.tr("Manual Test"))
        self.manual_test_button.setEnabled(False)
        self.manual_test_button.clicked.connect(self.on_manual_test_clicked)
        self.load_reference_button = QPushButton(qta.icon("mdi.map-check"), self.tr("Load Reference"))
        self.load_reference_button.clicked.connect(lambda: self.reference_view.load_dump(mark_ref=True))
        self.control_layout.addWidget(self.manual_test_button, 7, 0)
        self.control_layout.addWidget(self.load_reference_button, 7, 1)

        self.single_test_button = QPushButton(qta.icon("fa.play-circle"), self.tr("Single Test"))
        self.single_test_button.setEnabled(False)
        self.single_test_button.clicked.connect(self.on_single_test_clicked)
        self.continuous_test_button = QPushButton(qta.icon("mdi.playlist-play"), self.tr("Continuous Test"))
        self.continuous_test_button.setEnabled(False)
        self.continuous_test_button.clicked.connect(self.on_continuous_test_clicked)
        self.control_layout.addWidget(self.single_test_button, 8, 0)
        self.control_layout.addWidget(self.continuous_test_button, 8, 1)

        self.test_previous_button = QPushButton(qta.icon("mdi.skip-previous-circle"), self.tr("Test Previous"))
        self.test_previous_button.setEnabled(False)
        self.test_previous_button.clicked.connect(self.on_test_previous_clicked)
        self.test_next_button = QPushButton(qta.icon("mdi.skip-next-circle"), self.tr("Test Next"))
        self.test_next_button.setEnabled(False)
        self.test_next_button.clicked.connect(self.on_test_next_clicked)
        self.control_layout.addWidget(self.test_previous_button, 9, 0)
        self.control_layout.addWidget(self.test_next_button, 9, 1)

        # chart group
        self.chart_group = QGroupBox(self.tr("Chart"))
        self.chart_layout = QGridLayout(self.chart_group)
        self.result_chart = MixedDistributionChart(show_mode=True, toolbar=False)
        self.chart_layout.addWidget(self.result_chart, 0, 0)

        # table group
        self.table_group = QGroupBox(self.tr("Table"))
        self.reference_view = ReferenceResultViewer()
        self.result_view = FittingResultViewer(self.reference_view)
        self.result_view.result_marked.connect(lambda result: self.reference_view.add_references([result]))
        self.table_tab = QTabWidget()
        self.table_tab.addTab(self.result_view, qta.icon("fa.cubes"), self.tr("Result"))
        self.table_tab.addTab(self.reference_view, qta.icon("fa5s.key"), self.tr("Reference"))
        self.result_layout = QGridLayout(self.table_group)
        self.result_layout.addWidget(self.table_tab, 0, 0)

        # pack all group
        self.splitter1 = QSplitter(Qt.Orientation.Vertical)
        self.splitter1.addWidget(self.control_group)
        self.splitter1.addWidget(self.chart_group)
        self.splitter2 = QSplitter(Qt.Orientation.Horizontal)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.addWidget(self.table_group)
        self.main_layout.addWidget(self.splitter2, 0, 0)

    @property
    def distribution_type(self) -> DistributionType:
        distribution_type, _ = self.distribution_types[self.distribution_combo_box.currentIndex()]
        return distribution_type

    @property
    def n_components(self) -> int:
        return self.n_components_input.value()

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

    def on_load_dataset_clicked(self):
        self.load_dataset_dialog.show()

    def on_dataset_loaded(self, dataset: GrainSizeDataset):
        self.dataset = dataset
        self.n_samples_display.setText(str(dataset.n_samples))
        self.sample_index_input.setRange(1, dataset.n_samples)
        self.sample_index_input.setEnabled(True)
        self.manual_test_button.setEnabled(True)
        self.single_test_button.setEnabled(True)
        self.continuous_test_button.setEnabled(True)
        self.test_previous_button.setEnabled(True)
        self.test_next_button.setEnabled(True)

    def on_configure_fitting_clicked(self):
        if self.resolver_combo_box.currentText() == "classic":
            self.classic_setting.show()
        else:
            self.neural_setting.show()

    def on_sample_index_changed(self, index):
        self.sample_name_display.setText(self.dataset.samples[index-1].name)

    def generate_task(self, query_ref=True):
        sample_index = self.sample_index_input.value()-1
        sample = self.dataset.samples[sample_index]

        resolver = self.resolver_combo_box.currentText()
        if resolver == "classic":
            setting = self.classic_setting.setting
        else:
            setting = self.neural_setting.setting

        query = self.reference_view.query_reference(sample) # type: FittingResult
        if not query_ref or query is None:
            task = FittingTask(sample,
                               self.distribution_type,
                               self.n_components,
                               resolver=resolver,
                               resolver_setting=setting)
        else:
            keys = ["mean", "std", "skewness"]
            reference = [{key: comp.logarithmic_moments[key] for key in keys} for comp in query.components]
            task = FittingTask(sample,
                               query.distribution_type,
                               query.n_components,
                               resolver=resolver,
                               resolver_setting=setting,
                               reference=reference)
        return task

    def on_fitting_succeeded(self, fitting_result: FittingResult):
        # update chart
        self.result_chart.show_model(fitting_result.view_model)
        self.result_view.add_result(fitting_result)
        self.task_results[fitting_result.task.uuid] = fitting_result
        if self.__continuous_flag:
            if self.sample_index_input.value() == self.dataset.n_samples:
                self.on_continuous_test_clicked()
            else:
                self.sample_index_input.setValue(self.sample_index_input.value()+1)
                self.do_test()
                return
        self.manual_test_button.setEnabled(True)
        self.single_test_button.setEnabled(True)
        self.continuous_test_button.setEnabled(True)
        self.test_previous_button.setEnabled(True)
        self.test_next_button.setEnabled(True)

    def on_fitting_failed(self, failed_info: str, task: FittingTask):
        self.failed_task_ids.append(task.uuid)
        if self.__continuous_flag:
            self.on_continuous_test_clicked()
        self.manual_test_button.setEnabled(True)
        self.single_test_button.setEnabled(True)
        self.continuous_test_button.setEnabled(True)
        self.test_previous_button.setEnabled(True)
        self.test_next_button.setEnabled(True)
        self.show_error(failed_info)

    def do_test(self):
        self.manual_test_button.setEnabled(False)
        self.single_test_button.setEnabled(False)
        self.test_previous_button.setEnabled(False)
        self.test_next_button.setEnabled(False)
        if not self.__continuous_flag:
            self.continuous_test_button.setEnabled(False)
        task = self.generate_task()
        self.task_table[task.uuid] = task
        self.async_worker.execute_task(task)

    def on_manual_test_clicked(self):
        task = self.generate_task(query_ref=False)
        self.manual_panel.setup_task(task)
        self.manual_panel.show()

    def on_single_test_clicked(self):
        self.do_test()

    def on_continuous_test_clicked(self):
        if self.__continuous_flag:
            self.__continuous_flag = not self.__continuous_flag
            self.continuous_test_button.setText(self.tr("Continuous Test"))
        else:
            self.continuous_test_button.setText(self.tr("Cancel"))
            self.__continuous_flag = not self.__continuous_flag
            self.do_test()

    def on_test_previous_clicked(self):
        current = self.sample_index_input.value()
        if current == 1:
            return
        self.sample_index_input.setValue(current-1)
        self.do_test()

    def on_test_next_clicked(self):
        current = self.sample_index_input.value()
        if current == self.dataset.n_samples:
            return
        self.sample_index_input.setValue(current+1)
        self.do_test()

if __name__ == "__main__":
    import sys

    from QGrain.entry import setup_app
    app, splash = setup_app()
    main = SSUResolverPanel()
    # main.setWindowOpacity(0.95)
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
