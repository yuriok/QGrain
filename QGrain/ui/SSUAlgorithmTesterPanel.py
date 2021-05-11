from QGrain.ui.ReferenceResultViewer import ReferenceResultViewer
import logging
import typing

import numpy as np
import qtawesome as qta
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QComboBox, QDialog, QGridLayout, QGroupBox,
                               QLabel, QPushButton, QSpinBox, QSplitter, QTabWidget)
from QGrain.algorithms import DistributionType
from QGrain.algorithms.AsyncFittingWorker import AsyncFittingWorker
from QGrain.algorithms.moments import logarithmic
from QGrain.charts.MixedDistributionChart import MixedDistributionChart
from QGrain.models.artificial import ArtificialSample
from QGrain.models.FittingResult import FittingResult
from QGrain.models.FittingTask import FittingTask
from QGrain.ui.ClassicResolverSettingWidget import ClassicResolverSettingWidget
from QGrain.ui.FittingResultViewer import FittingResultViewer
from QGrain.ui.NNResolverSettingWidget import NNResolverSettingWidget
from QGrain.ui.RandomDatasetGenerator import RandomDatasetGenerator


class SSUAlgorithmTesterPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("Algorithm Tester"))
        self.distribution_types = [(DistributionType.Normal, self.tr("Normal")),
                                   (DistributionType.Weibull, self.tr("Weibull")),
                                   (DistributionType.SkewNormal, self.tr("Skew Normal"))]
        self.generate_setting = RandomDatasetGenerator(parent=self)
        self.classic_setting = ClassicResolverSettingWidget(parent=self)
        self.neural_setting = NNResolverSettingWidget(parent=self)
        self.async_worker = AsyncFittingWorker()
        self.async_worker.background_worker.task_succeeded.connect(self.on_fitting_succeeded)
        self.async_worker.background_worker.task_failed.connect(self.on_fitting_failed)
        self.task_table = {}
        self.task_results = {}
        self.failed_task_ids = []
        self.unquelified_task_ids = []
        self.__continuous_flag = False
        self.init_ui()

    def init_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        # control group
        self.control_group = QGroupBox(self.tr("Control"))
        self.control_layout = QGridLayout(self.control_group)
        self.resolver_label = QLabel(self.tr("Resolver"))
        self.resolver_combo_box = QComboBox()
        self.resolver_combo_box.addItems(["classic", "neural"])
        self.control_layout.addWidget(self.resolver_label, 0, 0)
        self.control_layout.addWidget(self.resolver_combo_box, 0, 1)
        self.configure_generating_button = QPushButton(qta.icon("fa.cubes"), self.tr("Configure Sample Generating"))
        self.configure_generating_button.clicked.connect(self.on_configure_generating_clicked)
        self.configure_fitting_button = QPushButton(qta.icon("fa.gears"), self.tr("Configure Fitting Algorithm"))
        self.configure_fitting_button.clicked.connect(self.on_configure_fitting_clicked)
        self.control_layout.addWidget(self.configure_generating_button, 1, 0)
        self.control_layout.addWidget(self.configure_fitting_button, 1, 1)
        self.distribution_label = QLabel(self.tr("Distribution Type"))
        self.distribution_combo_box = QComboBox()
        self.distribution_combo_box.addItems([name for _, name in self.distribution_types])
        self.component_number_label = QLabel(self.tr("Component Number"))
        self.n_components_input = QSpinBox()
        self.n_components_input.setRange(1, 10)
        self.n_components_input.setValue(3)
        self.control_layout.addWidget(self.distribution_label, 2, 0)
        self.control_layout.addWidget(self.distribution_combo_box, 2, 1)
        self.control_layout.addWidget(self.component_number_label, 3, 0)
        self.control_layout.addWidget(self.n_components_input, 3, 1)
        self.single_test_button = QPushButton(qta.icon("fa.play-circle"), self.tr("Single Test"))
        self.single_test_button.clicked.connect(self.on_single_test_clicked)
        self.continuous_test_button = QPushButton(qta.icon("mdi.playlist-play"), self.tr("Continuous Test"))
        self.continuous_test_button.clicked.connect(self.on_continuous_test_clicked)
        self.control_layout.addWidget(self.single_test_button, 4, 0)
        self.control_layout.addWidget(self.continuous_test_button, 4, 1)
        self.clear_stats_button = QPushButton(qta.icon("fa.eraser"), self.tr("Clear Statistics"))
        self.clear_stats_button.clicked.connect(self.clear_records)
        self.control_layout.addWidget(self.clear_stats_button, 5, 0, 1, 2)
        # chart group
        self.chart_group = QGroupBox(self.tr("Chart"))
        self.chart_layout = QGridLayout(self.chart_group)
        self.sample_chart = MixedDistributionChart(show_mode=True, toolbar=False)
        self.result_chart = MixedDistributionChart(show_mode=True, toolbar=False)
        self.chart_layout.addWidget(self.sample_chart, 0, 0)
        self.chart_layout.addWidget(self.result_chart, 0, 1)
        # stats group
        self.stats_group = QGroupBox(self.tr("Statistics"))
        self.stats_layout = QGridLayout(self.stats_group)
        self.n_task_label = QLabel(self.tr("Total Tasks:"))
        self.n_tasks_display = QLabel("0")
        self.n_failed_tasks_label = QLabel(self.tr("Failed Tasks:"))
        self.n_failed_tasks_display = QLabel("0")
        self.n_unqualified_tasks_label = QLabel(self.tr("Unqualified Tasks:"))
        self.n_unquelified_tasks_display = QLabel("0")
        self.stats_layout.addWidget(self.n_task_label, 0, 0)
        self.stats_layout.addWidget(self.n_tasks_display, 0, 1)
        self.stats_layout.addWidget(self.n_failed_tasks_label, 1, 0)
        self.stats_layout.addWidget(self.n_failed_tasks_display, 1, 1)
        self.stats_layout.addWidget(self.n_unqualified_tasks_label, 2, 0)
        self.stats_layout.addWidget(self.n_unquelified_tasks_display, 2, 1)
        self.mean_spent_time_label = QLabel(self.tr("Mean Spent Time [s]:"))
        self.mean_spent_time_display = QLabel("0.0")
        self.mean_n_iterations_label = QLabel(self.tr("Mean N<sub>iterations</sub>:"))
        self.mean_n_iterations_display = QLabel("0")
        self.mean_distance_label = QLabel(self.tr("Mean distance:"))
        self.mean_distance_display = QLabel("0.0")
        self.stats_layout.addWidget(self.mean_spent_time_label, 3, 0)
        self.stats_layout.addWidget(self.mean_spent_time_display, 3, 1)
        self.stats_layout.addWidget(self.mean_n_iterations_label, 4, 0)
        self.stats_layout.addWidget(self.mean_n_iterations_display, 4, 1)
        self.stats_layout.addWidget(self.mean_distance_label, 5, 0)
        self.stats_layout.addWidget(self.mean_distance_display, 5, 1)
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
        self.splitter1 = QSplitter(Qt.Orientation.Horizontal)
        self.splitter1.addWidget(self.control_group)
        self.splitter1.addWidget(self.stats_group)
        self.splitter2 = QSplitter(Qt.Orientation.Vertical)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.addWidget(self.chart_group)
        self.splitter3 = QSplitter(Qt.Orientation.Horizontal)
        self.splitter3.addWidget(self.splitter2)
        self.splitter3.addWidget(self.table_group)
        self.main_layout.addWidget(self.splitter3, 0, 0)

    @property
    def distribution_type(self) -> DistributionType:
        distribution_type, _ = self.distribution_types[self.distribution_combo_box.currentIndex()]
        return distribution_type

    @property
    def n_components(self) -> int:
        return self.n_components_input.value()

    def on_configure_generating_clicked(self):
        self.generate_setting.show()

    def on_configure_fitting_clicked(self):
        if self.resolver_combo_box.currentText() == "classic":
            self.classic_setting.show()
        else:
            self.neural_setting.show()

    def update_sample_chart(self, artificial_sample: ArtificialSample):
        self.sample_chart.show_model(artificial_sample.view_model)

    def update_fitting_chart(self, fitting_result: FittingResult):
        self.result_chart.show_model(fitting_result.view_model)

    def evaluate_result(self, artificial_sample: ArtificialSample, fitting_result: FittingResult, tolerance: float=0.1):
        component_errors = []
        unqualified = False
        for target, result in zip(artificial_sample.components, fitting_result.components):
            target_moments = logarithmic(artificial_sample.classes_φ, target.distribution)
            result_moments = logarithmic(artificial_sample.classes_φ, result.distribution)
            mean_error = np.abs((target_moments["mean"]-result_moments["mean"]) / target_moments["mean"])
            fraction_error = np.abs((target.fraction-result.fraction) / target.fraction)
            component_errors.append((mean_error, fraction_error))
            unqualified = (mean_error > tolerance) or (fraction_error > tolerance)
        return unqualified, component_errors

    def generate_task(self, query_ref=True):
        artificial_sample = self.generate_setting.get_random_sample()
        resolver = self.resolver_combo_box.currentText()
        if resolver == "classic":
            setting = self.classic_setting.setting
        else:
            setting = self.neural_setting.setting
        sample = artificial_sample.sample_to_fit
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
        return artificial_sample, task

    def update_stats(self):
        n_tasks = len(self.task_table)
        n_failed = len(self.failed_task_ids)
        n_unquelified = len(self.unquelified_task_ids)
        mean_spent_time = np.mean([result.time_spent for uuid, result in self.task_results.items()])
        mean_n_iterations = np.mean([result.n_iterations for uuid, result in self.task_results.items()])
        mean_distance = np.mean([result.get_distance(self.result_view.distance_name) for uuid, result in self.task_results.items()])
        self.n_tasks_display.setText(str(n_tasks))
        self.n_failed_tasks_display.setText(str(n_failed))
        self.n_unquelified_tasks_display.setText(str(n_unquelified))
        self.mean_spent_time_display.setText(f"{mean_spent_time:0.4f}")
        self.mean_n_iterations_display.setText(f"{mean_n_iterations:0.2f}")
        self.mean_distance_display.setText(f"{mean_distance:0.4f}")

    def on_fitting_succeeded(self, fitting_result: FittingResult):
        # update chart
        self.update_sample_chart(self.task_table[fitting_result.task.uuid][0])
        self.update_fitting_chart(fitting_result)
        self.task_results[fitting_result.task.uuid] = fitting_result
        self.result_view.add_result(fitting_result)
        if not fitting_result.is_valid:
            self.unquelified_task_ids.append(fitting_result.task.uuid)
        else:
            unqualified, errors = self.evaluate_result(self.task_table[fitting_result.task.uuid][0], fitting_result)
            if unqualified:
                self.unquelified_task_ids.append(fitting_result.task.uuid)

        self.update_stats()

        if self.__continuous_flag:
            self.do_test()
        self.single_test_button.setEnabled(True)
        self.continuous_test_button.setEnabled(True)
        self.clear_stats_button.setEnabled(True)

    def on_fitting_failed(self, failed_info: str, task: FittingTask):
        self.failed_task_ids.append(task.uuid)
        self.update_stats()
        if self.__continuous_flag:
            self.do_test()
        self.single_test_button.setEnabled(True)
        self.continuous_test_button.setEnabled(True)
        self.clear_stats_button.setEnabled(True)

    def clear_records(self):
        self.task_table = {}
        self.task_results = {}
        self.failed_task_ids = []
        self.unquelified_task_ids = []
        self.n_tasks_display.setText("0")
        self.n_failed_tasks_display.setText("0")
        self.n_unquelified_tasks_display.setText("0")
        self.mean_spent_time_display.setText("0.0")
        self.mean_n_iterations_display.setText("0")
        self.mean_distance_display.setText("0.0")

    def do_test(self):
        self.single_test_button.setEnabled(False)
        self.clear_stats_button.setEnabled(False)
        if not self.__continuous_flag:
            self.continuous_test_button.setEnabled(False)
        artificial_sample, task = self.generate_task()
        self.task_table[task.uuid] = (artificial_sample, task)
        self.async_worker.execute_task(task)

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


if __name__ == "__main__":
    import sys
    from QGrain.entry import setup_app
    app, splash = setup_app()
    main = SSUAlgorithmTesterPanel()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
