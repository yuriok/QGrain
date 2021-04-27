__all__ = ["ProcessState", "TaskWindow"]
import time
from math import sqrt
from typing import Dict, Iterable, List, Tuple
from uuid import UUID, uuid4

import numpy as np
from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import (QComboBox, QDialog, QGridLayout, QLabel,
                               QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QTableWidget, QWidget)

from QGrain.algorithms import DistributionType
from QGrain.models.FittingResult import FittingResult
from QGrain.models.SampleDataset import GrainSizeDataset
from QGrain.ui.AlgorithmSettingWidget import ClassicResolverSettingWidget


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
        # to calculate the residual time
        self.task_start_time = None
        self.task_accumulative_time = 0.0

        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.task_initialization_label = QLabel(self.tr("Task Initialization:"))
        self.task_initialization_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.task_initialization_label, 0, 0)

        self.sample_from_label = QLabel(self.tr("From"))
        self.sample_from_label.setToolTip(self.tr("Select the first sample you want to perform."))
        self.start_sample_combo_box = QComboBox()
        self.sample_to_label = QLabel(self.tr("To"))
        self.sample_to_label.setToolTip(self.tr("Select the last sample you want to perform."))
        self.end_sample_combo_box = QComboBox()
        self.main_layout.addWidget(self.sample_from_label, 1, 0)
        self.main_layout.addWidget(self.start_sample_combo_box, 1, 1)
        self.main_layout.addWidget(self.sample_to_label, 2, 0)
        self.main_layout.addWidget(self.end_sample_combo_box, 2, 1)

        self.interval_label = QLabel(self.tr("Interval"))
        self.interval_label.setToolTip(self.tr("Select the interval of each sample you want to perform."))
        self.interval_input = QSpinBox()
        self.interval_input.setRange(0, 9999)
        self.main_layout.addWidget(self.interval_label, 3, 0)
        self.main_layout.addWidget(self.interval_input, 3, 1)

        self.minimum_component_number_label = QLabel(self.tr("Minimum Component Number"))
        self.minimum_component_number_label.setToolTip(self.tr("Select the minimum component number you want to perform."))
        self.minimum_component_number = QSpinBox()
        self.minimum_component_number.setRange(1, 10)
        self.maximum_component_number_label = QLabel(self.tr("Maximum Component Number"))
        self.maximum_component_number_label.setToolTip(self.tr("Select the maximum component number you want to perform."))
        self.maximum_component_number = QSpinBox()
        self.maximum_component_number.setRange(1, 10)
        self.main_layout.addWidget(self.minimum_component_number_label, 4, 0)
        self.main_layout.addWidget(self.minimum_component_number, 4, 1)
        self.main_layout.addWidget(self.maximum_component_number_label, 5, 0)
        self.main_layout.addWidget(self.maximum_component_number, 5, 1)

        self.distribution_type_label = QLabel(self.tr("Distribution Type"))
        self.distribution_type_label.setToolTip(self.tr("Select the base distribution function of each component."))
        self.distribution_type_combo_box = QComboBox()
        self.distribution_type_options = {self.tr("Normal"): DistributionType.Normal,
                                          self.tr("Weibull"): DistributionType.Weibull,
                                          self.tr("Gen. Weibull"): DistributionType.GeneralWeibull}
        self.distribution_type_combo_box.addItems(self.distribution_type_options.keys())
        self.distribution_type_combo_box.setCurrentIndex(2)
        self.main_layout.addWidget(self.distribution_type_label, 6, 0)
        self.main_layout.addWidget(self.distribution_type_combo_box, 6, 1)

        self.algorithm_setting_widget = ClassicResolverSettingWidget()
        self.algorithm_setting_widget.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.algorithm_setting_widget, 7, 0, 1, 2)

        self.generate_task_button = QPushButton(self.tr("Generate Tasks"))
        self.generate_task_button.setToolTip(self.tr("Click to generate the fitting tasks."))
        self.main_layout.addWidget(self.generate_task_button, 8, 0, 1, 2)

        self.process_state_label = QLabel(self.tr("Process State:"))
        self.process_state_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.process_state_label, 9, 0)

        self.not_started_label = QLabel(self.tr("Not Started"))
        self.not_started_label.setToolTip(self.tr("The number of not started tasks."))
        self.not_started_display = QLabel("0")
        self.main_layout.addWidget(self.not_started_label, 10, 0)
        self.main_layout.addWidget(self.not_started_display, 10, 1)

        self.succeeded_label = QLabel(self.tr("Succeeded"))
        self.succeeded_label.setToolTip(self.tr("The number of succeeded tasks."))
        self.succeeded_display = QLabel("0")
        self.main_layout.addWidget(self.succeeded_label, 11, 0)
        self.main_layout.addWidget(self.succeeded_display, 11, 1)

        self.failed_label = QLabel(self.tr("Failed"))
        self.failed_label.setToolTip(self.tr("The number of failed tasks."))
        self.failed_display = QLabel("0")
        self.main_layout.addWidget(self.failed_label, 12, 0)
        self.main_layout.addWidget(self.failed_display, 12, 1)

        self.time_spent_label = QLabel(self.tr("Time Spent"))
        self.time_spent_label.setToolTip(self.tr("The spent time of these fitting tasks."))
        self.time_spent_dispaly = QLabel("0:00:00")
        self.time_left_label = QLabel(self.tr("Time Left"))
        self.time_left_label.setToolTip(self.tr("The left time of these fitting tasks."))
        self.time_left_display = QLabel("99:59:59")
        self.main_layout.addWidget(self.time_spent_label, 13, 0)
        self.main_layout.addWidget(self.time_spent_dispaly, 13, 1)
        self.main_layout.addWidget(self.time_left_label, 14, 0)
        self.main_layout.addWidget(self.time_left_display, 14, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)
        self.main_layout.addWidget(self.progress_bar, 15, 0, 1, 2)

        self.run_button = QPushButton(self.tr("Run"))
        self.run_button.setToolTip(self.tr("Click to run / pause these fitting tasks."))
        self.run_button.setEnabled(False)
        self.finish_button = QPushButton(self.tr("Finish"))
        self.finish_button.setToolTip(self.tr("Click to finish these fitting progress, record the succeeded results."))
        self.finish_button.setEnabled(False)
        self.main_layout.addWidget(self.run_button, 16, 0)
        self.main_layout.addWidget(self.finish_button, 16, 1)

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
        interval = self.interval_input.value() + 1
        # make sure the order is positive
        if start > end:
            start, end = end, start
        return self.grain_size_data.samples[start: end+1: interval]

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

    def on_data_loaded(self, data: GrainSizeDataset):
        if data is None:
            return
        elif not data.has_data:
            return
        self.grain_size_data = data
        sample_names = [sample.name for sample in data.samples]
        self.start_sample_combo_box.clear()
        self.start_sample_combo_box.addItems(sample_names)
        self.start_sample_combo_box.setCurrentIndex(0)
        self.end_sample_combo_box.clear()
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
        # update the ui
        self.time_left_display.setText("99:59:59")
        new_task_number = len(tasks)
        all_task_number = new_task_number + self.progress_bar.maximum()
        self.progress_bar.setMaximum(all_task_number)
        not_started_number = int(self.not_started_display.text()) + new_task_number
        self.not_started_display.setText(str(not_started_number))
        self.run_button.setEnabled(True)
        self.finish_button.setEnabled(True)

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

        # update the ui
        self.not_started_display.setText(str(not_started_task_number))
        self.succeeded_display.setText(str(succeeded_task_number))
        self.failed_display.setText(str(failed_task_number))
        self.progress_bar.setMaximum(task_number)
        self.progress_bar.setValue(succeeded_task_number + failed_task_number)
        # calculate the spent and left time of tasks
        if self.task_start_time is not None:
            time_spent = time.time() - self.task_start_time + self.task_accumulative_time
        else:
            time_spent = self.task_accumulative_time
        if not_started_task_number == task_number:
            time_left = 359999 # equals to 99:59:59
        elif not_started_task_number == 0:
            time_left = 0
        else:
            time_left = time_spent / (succeeded_task_number + failed_task_number) * not_started_task_number
        def second_to_hms(seconds: float):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}"
        self.time_spent_dispaly.setText(second_to_hms(time_spent))
        self.time_left_display.setText(second_to_hms(time_left))

        self.tasks = tasks
        self.states = states
        self.succeeded_results = succeeded_results

        if not_started_task_number == 0:
            self.running_flag = False
            self.task_accumulative_time += time.time() - self.task_start_time
            self.task_start_time = None
            self.generate_task_button.setEnabled(True)
            self.run_button.setText(self.tr("Run"))
            self.run_button.setEnabled(False)
            self.finish_button.setEnabled(True)

    def on_run_button_clicked(self):
        if not self.running_flag:
            self.running_flag = True
            self.task_start_time = time.time()
            self.fitting_started_signal.emit()
            self.generate_task_button.setEnabled(False)
            self.run_button.setEnabled(True)
            self.run_button.setText(self.tr("Pause"))
            self.finish_button.setEnabled(False)
        else:
            if self.multiprocessing_resolver is not None:
                self.multiprocessing_resolver.pause_task()
            self.running_flag = False
            self.task_accumulative_time += time.time() - self.task_start_time
            self.task_start_time = None
            self.generate_task_button.setEnabled(True)
            self.run_button.setEnabled(True)
            self.run_button.setText(self.tr("Run"))
            self.finish_button.setEnabled(True)

    def cleanup(self):
        if self.multiprocessing_resolver is not None:
            self.multiprocessing_resolver.cleanup()
        self.tasks = None
        self.states = None
        self.succeeded_results = None
        self.task_start_time = None
        self.task_accumulative_time = 0.0
        self.not_started_display.setText("0")
        self.succeeded_display.setText("0")
        self.failed_display.setText("0")
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)
        self.time_spent_dispaly.setText("0:00:00")
        self.time_left_display.setText("99:59:59")
        self.run_button.setText(self.tr("Run"))
        self.running_flag = False
        self.generate_task_button.setEnabled(True)
        self.run_button.setEnabled(False)
        self.finish_button.setEnabled(False)

    def check_result(self):
        # classify the results by samples' id
        results_by_sample_id = {}
        for task in self.tasks:
            if self.states[task.uuid] == ProcessState.Succeeded:
                if task.sample.uuid in results_by_sample_id:
                    results_by_sample_id[task.sample.uuid].append(self.succeeded_results[task.uuid])
                else:
                    results_by_sample_id[task.sample.uuid] = [self.succeeded_results[task.uuid]]

        checked_results = []
        for sample_id, results in results_by_sample_id.items():
            valid_results = {}
            mse_values = {}
            minimum_mse_values = {}
            optional_component_numbers = {}
            for result in results:
                # record the mean squared errors to judge if the component number of the small-end sample is not enough
                mse_values[result.component_number] = result.mean_squared_error
                # check if there is any needless component
                has_needless_component = False
                needless_component_number = 0
                for component in result.components:
                    # if the fraction of any component is less than 0.01%
                    if component.fraction < 1e-4 or component.has_nan:
                        has_needless_component = True
                        needless_component_number += 1
                # if there is any needless component, the mean squared error is close to the lowest level of this sample
                if has_needless_component:
                    minimum_mse_values[result.component_number] = result.mean_squared_error
                    optional_component_numbers[result.component_number] = result.component_number - needless_component_number
                    continue
                # ignore invalid result
                if result.has_invalid_value:
                    continue
                valid_results[result.component_number] = result

            optional_component_number_count = {}
            for component_number, optional_component_number in optional_component_numbers.items():
                if optional_component_number in optional_component_number_count:
                    optional_component_number_count[optional_component_number] += 1
                else:
                    optional_component_number_count[optional_component_number] = 1

            min_count_component_number = None
            min_count = 10000
            for optional_component_number, count in optional_component_number_count.items():
                if count < min_count:
                    min_count = count
                    min_count_component_number = optional_component_number
            # for result in results:
            #     if result.component_number == min_count_component_number:
            #         checked_results.append(result)
            #         break

            # calculate the 1/4, 1/2, and 3/4 postion value to judge which result is invalid
            # 1. the mean squared errors are much higher in the results which are lack of components
            # 2. with the component number getting higher, the mean squared error will get lower and finally reach the minimum
            median = np.median(list(mse_values.values()))
            upper_group = [value for value in mse_values.values() if value >= median]
            lower_group = [value for value in mse_values.values() if value <= median]
            value_1_4 = np.median(lower_group)
            value_3_4 = np.median(upper_group)
            distance_QR = value_3_4 - value_1_4
            non_outlier_results = []
            for component_number, result in valid_results.items():
                if np.abs(result.mean_squared_error - median) < distance_QR * 2.5:
                    non_outlier_results.append(result)
            if len(non_outlier_results) >= 1:
                least_result = non_outlier_results[0]
                for result in non_outlier_results:
                    if result.component_number < least_result.component_number:
                        least_result = result
                checked_results.append(least_result)
            else:
                for result in results:
                    if result.component_number == min_count_component_number:
                        checked_results.append(result)
                        break

        self.fitting_finished_signal.emit(checked_results)


    def on_finish_button_clicked(self):
        if self.tasks is None:
            assert self.states is None
            assert self.succeeded_results is None
        else:
            # self.check_result()
            not_started_number = int(self.not_started_display.text())
            self.fitting_finished_signal.emit(list(self.succeeded_results.values()))
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
        else:
            e.accept()


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)
    task_window = TaskWindow()
    task_window.show()
    sys.exit(app.exec_())
