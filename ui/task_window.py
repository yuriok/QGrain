from enum import Enum, unique
from math import sqrt
from typing import Iterable, Tuple

from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
                               QSizePolicy, QWidget)

from data import FittingResult
from resolvers import FittingTask


@unique
class ProcessState(Enum):
    Unknown = 0
    Succeeded = 1
    Failed = -1


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

        self.state_labels_widget = QWidget()
        self.main_layout.addWidget(self.state_labels_widget, 0, 0)
        self.state_labels_layout = QGridLayout(self.state_labels_widget)
        

        self.setWindowTitle(self.tr("The States of Fitting Tasks"))
        # self.setWindowFlags(Qt.Drawer | Qt.WindowStaysOnTopHint)
        self.setWindowFlags(Qt.Drawer)

    def change_label_state(self, label: QLabel, state: ProcessState):
        if state == ProcessState.Unknown:
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
        base_number = sqrt(task_number/(self.WIDTH_WEIGHT*self.HEIGHT_WEIGHT))
        column_number = base_number*self.WIDTH_WEIGHT // 1 + 1

        for i, task in enumerate(tasks):
            label = QLabel()
            label.setFixedSize(16, 16)
            self.change_label_state(label, ProcessState.Unknown)
            label.setToolTip(task.sample_name)
            row = i // column_number
            col = i % column_number
            self.state_labels_layout.addWidget(label, row, col)
            self.task_state_labels.update({task.sample_id: label})

    def on_task_state_updated(self, states: Iterable[Tuple]):
        self.show()
        for sample_id, succeeded in states:
            if succeeded:
                label = self.task_state_labels[sample_id]
                self.change_label_state(label, ProcessState.Succeeded)

    def on_task_finished(self, succeeded_results: Iterable[FittingResult], failed_tasks: Iterable[FittingTask]):
        for task in failed_tasks:
            label = self.task_state_labels[task.sample_id]
            self.change_label_state(label, ProcessState.Failed)