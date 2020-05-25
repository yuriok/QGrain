__all__ = ["ProcessState", "TaskWindow"]
from math import sqrt
from typing import Dict, Iterable, List, Tuple
from uuid import UUID, uuid4

from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import (QComboBox, QDialog, QGridLayout, QLabel,
                               QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QTableWidget, QWidget)

from QGrain.algorithms import DistributionType
from QGrain.models.FittingResult import FittingResult
from QGrain.models.SampleDataset import SampleDataset
from QGrain.resolvers.HeadlessResolver import FittingTask
from QGrain.resolvers.MultiprocessingResolver import (MultiProcessingResolver,
                                                      ProcessState)
from QGrain.ui.AlgorithmSettingWidget import AlgorithmSettingWidget

import numpy as np


class TaskWindow(QDialog):
    task_generated_signal = Signal(list)
    fitting_started_signal = Signal()
    fitting_finished_signal = Signal(list)

    def __init__(self, parent=None,
                 multiprocessing_resolver: MultiProcessingResolver = None):
        super().__init__(parent=parent)
        self.multiprocessing_resolver = multiprocessing_resolver
        self.grain_size_data = None  # type: SampleDataset
        self.running_flag = False
        self.tasks = None
        self.states = None
        self.succeeded_results = None
        self.staging_tasks = None
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.task_initialization_label = QLabel(self.tr("Task Initialization:"))
        self.task_initialization_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.task_initialization_label, 0, 0)

        self.sample_from_label = QLabel(self.tr("From"))
        self.start_sample_combo_box = QComboBox()
        self.sample_to_label = QLabel(self.tr("To"))
        self.end_sample_combo_box = QComboBox()
        self.main_layout.addWidget(self.sample_from_label, 1, 0)
        self.main_layout.addWidget(self.start_sample_combo_box, 1, 1)
        self.main_layout.addWidget(self.sample_to_label, 2, 0)
        self.main_layout.addWidget(self.end_sample_combo_box, 2, 1)

        self.minimum_component_number_label = QLabel(self.tr("Minimum Component Number"))
        self.minimum_component_number = QSpinBox()
        self.minimum_component_number.setRange(1, 10)
        self.maximum_component_number_label = QLabel(self.tr("Maximum Component Number"))
        self.maximum_component_number = QSpinBox()
        self.maximum_component_number.setRange(1, 10)
        self.main_layout.addWidget(self.minimum_component_number_label, 3, 0)
        self.main_layout.addWidget(self.minimum_component_number, 3, 1)
        self.main_layout.addWidget(self.maximum_component_number_label, 4, 0)
        self.main_layout.addWidget(self.maximum_component_number, 4, 1)

        self.distribution_type_label = QLabel(self.tr("Distribution Type"))
        self.distribution_type_combo_box = QComboBox()
        self.distribution_type_options = {self.tr("Normal"): DistributionType.Normal,
                                          self.tr("Weibull"): DistributionType.Weibull,
                                          self.tr("Gen. Weibull"): DistributionType.GeneralWeibull}
        self.distribution_type_combo_box.addItems(self.distribution_type_options.keys())
        self.distribution_type_combo_box.setCurrentIndex(2)
        self.main_layout.addWidget(self.distribution_type_label, 5, 0)
        self.main_layout.addWidget(self.distribution_type_combo_box, 5, 1)

        self.algorithm_setting_widget = AlgorithmSettingWidget()
        self.algorithm_setting_widget.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.algorithm_setting_widget, 6, 0, 1, 2)

        self.generate_task_button = QPushButton(self.tr("Generate Tasks"))
        self.main_layout.addWidget(self.generate_task_button, 7, 0, 1, 2)

        self.process_state_label = QLabel(self.tr("Process State:"))
        self.process_state_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.process_state_label, 8, 0)

        self.not_started_label = QLabel(self.tr("Not Started"))
        self.not_started_display = QLabel("0")
        self.main_layout.addWidget(self.not_started_label, 9, 0)
        self.main_layout.addWidget(self.not_started_display, 9, 1)

        self.succeeded_label = QLabel(self.tr("Succeeded"))
        self.succeeded_display = QLabel("0")
        self.main_layout.addWidget(self.succeeded_label, 10, 0)
        self.main_layout.addWidget(self.succeeded_display, 10, 1)

        self.failed_label = QLabel(self.tr("Failed"))
        self.failed_display = QLabel("0")
        self.main_layout.addWidget(self.failed_label, 11, 0)
        self.main_layout.addWidget(self.failed_display, 11, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)
        self.main_layout.addWidget(self.progress_bar, 12, 0, 1, 2)

        self.run_button = QPushButton(self.tr("Run"))
        self.finish_button = QPushButton(self.tr("Finish"))
        self.main_layout.addWidget(self.run_button, 13, 0)
        self.main_layout.addWidget(self.finish_button, 13, 1)

        self.generate_task_button.clicked.connect(self.on_generate_task_button_clicked)
        self.run_button.clicked.connect(self.on_run_button_clicked)
        self.finish_button.clicked.connect(self.on_finish_button_clicked)

        self.setWindowTitle(self.tr("The States of Fitting Tasks"))
        self.setWindowFlags(Qt.Drawer)

        self.cancel_msg_box = QMessageBox(self)
        self.cancel_msg_box.setWindowTitle(self.tr("Warning"))
        self.cancel_msg_box.setText(self.tr("Are you sure to cancel the tasks?"))
        self.cancel_msg_box.addButton(QMessageBox.StandardButton.Yes)
        self.cancel_msg_box.addButton(QMessageBox.StandardButton.No)
        self.cancel_msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        self.cancel_msg_box.setWindowFlags(Qt.Drawer)

        self.storage_msg_box = QMessageBox(self)
        self.storage_msg_box.setWindowTitle(self.tr("Info"))
        self.storage_msg_box.setText(self.tr("Storage the left and failed tasks for the next processing?"))
        self.storage_msg_box.addButton(QMessageBox.StandardButton.Yes)
        self.storage_msg_box.addButton(QMessageBox.StandardButton.No)
        self.storage_msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
        self.storage_msg_box.setWindowFlags(Qt.Drawer)

    @property
    def samples(self):
        if self.grain_size_data is None:
            return []
        start = self.start_sample_combo_box.currentIndex()
        end = self.end_sample_combo_box.currentIndex()
        # make sure the order is positive
        if start > end:
            start, end = end, start
        return self.grain_size_data.samples[start: end+1]

    @property
    def distribution_type(self):
        return self.distribution_type_options[self.distribution_type_combo_box.currentText()]

    @property
    def component_numbers(self):
        min_number = self.minimum_component_number.value()
        max_number = self.maximum_component_number.value()
        # make sure the order is positive
        if min_number > max_number:
            max_number, max_number = max_number, max_number
        return list(range(min_number, max_number+1))

    def on_data_loaded(self, data: SampleDataset):
        if data is None:
            return
        elif not data.has_data:
            return
        self.grain_size_data = data
        sample_names = [sample.name for sample in data.samples]
        self.start_sample_combo_box.addItems(sample_names)
        self.start_sample_combo_box.setCurrentIndex(0)
        self.end_sample_combo_box.addItems(sample_names)
        self.end_sample_combo_box.setCurrentIndex(len(sample_names)-1)

    def on_generate_task_button_clicked(self):
        tasks = []
        if self.staging_tasks is not None:
            tasks.extend(self.staging_tasks)
            self.staging_tasks = None
        for sample in self.samples:
            for component_number in self.component_numbers:
                task = FittingTask(
                    sample,
                    component_number=component_number,
                    distribution_type=self.distribution_type,
                    algorithm_settings=self.algorithm_setting_widget.algorithm_settings)
                tasks.append(task)
        self.task_generated_signal.emit(tasks)

        new_task_number = len(tasks)
        all_task_number = new_task_number + self.progress_bar.maximum()
        self.progress_bar.setMaximum(all_task_number)
        not_started_number = int(self.not_started_display.text()) + new_task_number
        self.not_started_display.setText(str(not_started_number))

    def on_task_state_updated(self, tasks: List[FittingTask],
                              states: Dict[UUID, ProcessState],
                              succeeded_results: Dict[UUID, FittingResult]):
        assert tasks is not None
        assert states is not None
        assert succeeded_results is not None

        task_number = len(tasks)
        succeeded_task_number = len([value for value in states.values() if value == ProcessState.Succeeded])
        failed_task_number = len([value for value in states.values() if value == ProcessState.Failed])
        not_started_task_number = task_number - succeeded_task_number - failed_task_number

        self.not_started_display.setText(str(not_started_task_number))
        self.succeeded_display.setText(str(succeeded_task_number))
        self.failed_display.setText(str(failed_task_number))
        self.progress_bar.setMaximum(task_number)
        self.progress_bar.setValue(succeeded_task_number + failed_task_number)

        self.tasks = tasks
        self.states = states
        self.succeeded_results = succeeded_results


    def on_run_button_clicked(self):
        if not self.running_flag:
            self.running_flag = True
            self.fitting_started_signal.emit()
            self.run_button.setText(self.tr("Pause"))
            self.finish_button.setEnabled(False)
        else:
            if self.multiprocessing_resolver is not None:
                self.multiprocessing_resolver.pause_task()
            self.running_flag = False
            self.run_button.setText(self.tr("Run"))
            self.finish_button.setEnabled(True)

    def cleanup(self):
        if self.multiprocessing_resolver is not None:
            self.multiprocessing_resolver.cleanup()
        self.not_started_display.setText("0")
        self.succeeded_display.setText("0")
        self.failed_display.setText("0")
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)
        self.tasks = None
        self.states = None
        self.succeeded_results = None
        self.run_button.setText(self.tr("Run"))
        self.running_flag = False
        self.run_button.setEnabled(True)
        self.finish_button.setEnabled(False)


    def check_result(self):
        sorted_by_sample = {}
        for task_id, result in self.succeeded_results.items():
            if result.name in sorted_by_sample:
                sorted_by_sample[result.name].append(result)
            else:
                sorted_by_sample[result.name] = [result]

        checked_results = []
        for sample_name, results in sorted_by_sample.items():
            valid_results = []
            mse_values = []
            for result in results:
                if result.has_invalid_value:
                    continue
                has_zero_component = False
                
                for component in result.components:
                    if component.fraction < 1e-4:
                        has_zero_component = True
                        break
                if has_zero_component:
                    mse_values.append(result.mean_squared_error)
                    continue
                valid_results.append(result)

            valid_results2 = []

            for result in valid_results:
                mean = np.mean(mse_values)
                f = np.abs(result.mean_squared_error - mean) / mean
                if f < 5:
                    valid_results2.append(result)

            least_result = valid_results2[0]
            for result in valid_results2:
                if result.component_number < least_result.component_number:
                    least_result = result
            checked_results.append(least_result)

        self.fitting_finished_signal.emit(checked_results)


    def on_finish_button_clicked(self):
        not_started_number = int(self.not_started_display.text())
        self.fitting_finished_signal.emit(list(self.succeeded_results.values()))
        # self.check_result()
        if not_started_number != 0:
            res = self.storage_msg_box.exec_()
            if res == QMessageBox.Yes:
                staging_tasks = []
                for task in self.tasks:
                    if self.states[task.uuid] != ProcessState.Succeeded:
                        staging_tasks.append(task)
                self.staging_tasks = staging_tasks
        self.cleanup()

    def closeEvent(self, e):
        if self.running_flag:
            res = self.cancel_msg_box.exec_()
            if res == QMessageBox.Yes:
                self.on_run_button_clicked()
                e.accept()
            else:
                e.ignore()
        self.cleanup()


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)
    task_window = TaskWindow()
    task_window.show()
    sys.exit(app.exec_())
