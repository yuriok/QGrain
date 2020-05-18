__all__ = ["ProcessState", "TaskWindow"]
from math import sqrt
from typing import Dict, Iterable, Tuple

from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QGridLayout, QLabel, QListWidget, QMainWindow,
                               QWidget)

from QGrain.models.FittingResult import FittingResult
from QGrain.resolvers.HeadlessResolver import FittingTask
from QGrain.resolvers.MultiprocessingResolver import ProcessState


class TaskWindow(QMainWindow):
    WIDTH_WEIGHT = 16
    HEIGHT_WEIGHT = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()
        self.task_state_labels = {}

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout(self.central_widget)

        self.not_started_label = QLabel(self.tr("Not Started:"))
        self.not_started_display = QLabel(self.tr("Unknown:"))
        self.main_layout.addWidget(self.not_started_label, 0, 0)
        self.main_layout.addWidget(self.not_started_display, 0, 1)

        self.succeeded_label = QLabel(self.tr("Succeeded:"))
        self.succeeded_display = QLabel(self.tr("Unknown"))
        self.main_layout.addWidget(self.succeeded_label, 0, 2)
        self.main_layout.addWidget(self.succeeded_display, 0, 3)

        self.failed_label = QLabel(self.tr("Failed:"))
        self.failed_display = QLabel(self.tr("Unknown"))
        self.main_layout.addWidget(self.failed_label, 0, 4)
        self.main_layout.addWidget(self.failed_display, 0, 5)

        self.state_labels_widget = QWidget()
        self.main_layout.addWidget(self.state_labels_widget, 1, 0, 1, 6)
        self.state_labels_layout = QGridLayout(self.state_labels_widget)

        self.setWindowTitle(self.tr("The States of Fitting Tasks"))
        self.setWindowFlags(Qt.Drawer)

    def change_label_state(self, label: QLabel, state: ProcessState):
        if state == ProcessState.NotStarted:
            label.setStyleSheet("QLabel{background:#FFE140;}")
        elif state == ProcessState.Succeeded:
            label.setStyleSheet("QLabel{background:#00A779;}")
        elif state == ProcessState.Failed:
            label.setStyleSheet("QLabel{background:#ED002F;}")
        else:
            raise NotImplementedError(state)

    def on_task_initialized(self, tasks: Iterable[FittingTask]):
        for i, label in self.task_state_labels.items():
            self.state_labels_layout.removeWidget(label)
        self.task_state_labels.clear()

        task_number = len(tasks)
        # row number = base_number*HEIGHT_WEIGHT
        # column number = base_number*WIDTH_WEIGHT
        base_number = sqrt(task_number / (self.WIDTH_WEIGHT*self.HEIGHT_WEIGHT))
        column_number = base_number*self.WIDTH_WEIGHT // 1 + 1

        for i, task in enumerate(tasks):
            label = QLabel()
            label.setFixedSize(16, 16)
            self.change_label_state(label, ProcessState.NotStarted)
            label.setToolTip(task.sample.name)
            row = i // column_number
            col = i % column_number
            self.state_labels_layout.addWidget(label, row, col)
            self.task_state_labels.update({task.sample.uuid: label})

    def on_task_state_updated(self, states: Dict):
        for sample_id, state in states.items():
            label = self.task_state_labels[sample_id]
            self.change_label_state(label, state)

        task_number = len(states)
        succeeded_task_number = len([value for value in states.values() if value == ProcessState.Succeeded])
        failed_task_number = len([value for value in states.values() if value == ProcessState.Failed])
        not_started_task_number = task_number - succeeded_task_number - failed_task_number

        self.not_started_display.setText(str(not_started_task_number))
        self.succeeded_display.setText(str(succeeded_task_number))
        self.failed_display.setText(str(failed_task_number))

    def on_task_finished(self, succeeded_results: Iterable[FittingResult], failed_tasks: Iterable[FittingTask]):
        for task in failed_tasks:
            label = self.task_state_labels[task.sample_id]
            self.change_label_state(label, ProcessState.Failed)



if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)
    task_window = TaskWindow()
    task_window.show()
    # task_window.on_task_initialized([str(i) for i in range(100)])
    sys.exit(app.exec_())
