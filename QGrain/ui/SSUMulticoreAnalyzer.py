
import multiprocessing as mp
import typing
from enum import Enum, unique
from uuid import UUID

import psutil
from PySide6 import QtCore, QtGui, QtWidgets

from ..chart.config_matplotlib import backgroud_color
from ..io import GrainSizeDataset
from ..ssu import BasicResolver, DistributionType, SSUResult, SSUTask


@unique
class TaskState(Enum):
    NotStarted = 0
    Processing = 1
    Failed = 2
    Finished = 3


def excute_tasks(
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        failed_queue: mp.Queue,
        event_queue: mp.Queue,
        retry_times: int = 5):
    resolver = BasicResolver()
    while True:
        task = task_queue.get()
        left_times = retry_times
        event_queue.put((task, TaskState.Processing))
        while left_times > 0:
            message, result = resolver.try_fit(task)
            if isinstance(result, SSUResult):
                result_queue.put(result)
                break
            else:
                left_times -= 1
        if left_times > 0:
            event_queue.put((task, TaskState.Finished))
        else:
            failed_queue.put(task)
            event_queue.put((task, TaskState.Failed))


class TaskStateBubble(QtWidgets.QWidget):
    def __init__(self, size=16, border_radius=4):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.__size = size
        self.__border_radius = border_radius
        self.init_ui()

    def get_qss(self, state: TaskState):
        if state == TaskState.NotStarted:
            return f"border-radius: {self.__border_radius}px;border:0px;background:#7a7374"
        elif state == TaskState.Processing:
            return f"border-radius: {self.__border_radius}px;border:0px;background:#ff9900"
        elif state == TaskState.Failed:
            return f"border-radius: {self.__border_radius}px;border:0px;background:#36292f"
        elif state == TaskState.Finished:
            return f"border-radius: {self.__border_radius}px;border:0px;background:#8abcd1"
        else:
            raise NotImplementedError(state)

    def change_state(self, state: TaskState):
        qss = self.get_qss(state)
        self.display.setStyleSheet(qss)

    def init_ui(self):
        self.main_layout = QtWidgets.QGridLayout(self)
        self.display = QtWidgets.QLabel()
        self.display.setFixedSize(self.__size, self.__size)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.display)
        self.change_state(TaskState.NotStarted)

    def make_transparent(self):
        sheet = f"border-radius: {self.__border_radius}px;border:0px;background:{backgroud_color()}"
        self.display.setStyleSheet(sheet)


class SSUMulticoreAnalyzer(QtWidgets.QDialog):
    BUBBLE_ROWS = 10
    BUBBLE_COLUMNS = 20
    result_finished = QtCore.Signal(SSUResult)
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.__task_queue = mp.Queue()
        self.__result_queue = mp.Queue()
        self.__failed_queue = mp.Queue()
        self.__event_queue = mp.Queue()
        self.__tasks = [] # type: list[SSUTask]
        self.__index_map = {} # type: dict[UUID, int]
        self.__state_map = {} # type: dict[int, TaskState]
        self.__results = {} # type: dict[int, SSUResult]
        self.__failed_tasks = {} # type: dict[int, SSUTask]
        self.__processes = [] # type: list[mp.Process]
        self.__start_flag = False
        self.__watcher = QtCore.QTimer()
        self.__watcher.setSingleShot(False)
        self.__watcher.setInterval(10)
        self.__watcher.timeout.connect(self.watch)
        self.init_ui()

    @property
    def n_tasks(self) -> int:
        return len(self.__tasks)

    @property
    def page_size(self) -> int:
        return self.BUBBLE_ROWS * self.BUBBLE_COLUMNS

    @property
    def n_pages(self) -> int:
        n_pages, left = divmod(self.n_tasks, self.page_size)
        if left > 0:
            n_pages += 1
        return n_pages

    @property
    def page_index(self) -> int:
        return self.page_combo_box.currentIndex()

    def in_page(self, index: int) -> bool:
        return self.page_size*self.page_index <= index < self.page_size * (self.page_index + 1)

    def init_ui(self):
        self.setWindowTitle(self.tr("SSU Multicore Analyzer"))
        self.main_layout = QtWidgets.QGridLayout(self)
        self.control_group = QtWidgets.QGroupBox(self.tr("Control"))
        self.control_layout = QtWidgets.QGridLayout(self.control_group)
        self.n_tasks_label = QtWidgets.QLabel(self.tr("Total Number of Tasks"))
        self.n_tasks_display = QtWidgets.QLabel(self.tr("Unknown"))
        self.n_workers_label = QtWidgets.QLabel(self.tr("Number of Workers"))
        self.n_workers_input = QtWidgets.QSpinBox()
        self.n_workers_input.setRange(1, psutil.cpu_count(logical=False))
        self.n_workers_input.setValue(2)
        self.n_remains_label = QtWidgets.QLabel(self.tr("Number of Remaining Tasks"))
        self.n_remains_display = QtWidgets.QLabel(self.tr("Unknown"))
        self.n_finished_tasks_label = QtWidgets.QLabel(self.tr("Number of Finished Tasks"))
        self.n_finished_tasks_display = QtWidgets.QLabel(self.tr("Unknown"))
        self.n_failed_tasks_label = QtWidgets.QLabel(self.tr("Number of Failed Tasks"))
        self.n_failed_tasks_display = QtWidgets.QLabel(self.tr("Unknown"))
        self.run_button = QtWidgets.QPushButton(self.tr("Start"))
        self.run_button.clicked.connect(self.on_run_button_clicked)
        self.control_layout.addWidget(self.n_tasks_label, 0, 0)
        self.control_layout.addWidget(self.n_tasks_display, 0, 1)
        self.control_layout.addWidget(self.n_workers_label, 0, 2)
        self.control_layout.addWidget(self.n_workers_input, 0, 3)
        self.control_layout.addWidget(self.n_remains_label, 1, 0)
        self.control_layout.addWidget(self.n_remains_display, 1, 1)
        self.control_layout.addWidget(self.n_finished_tasks_label, 1, 2)
        self.control_layout.addWidget(self.n_finished_tasks_display, 1, 3)
        self.control_layout.addWidget(self.n_failed_tasks_label, 2, 0)
        self.control_layout.addWidget(self.n_failed_tasks_display, 2, 1)
        self.control_layout.addWidget(self.run_button, 2, 2, 1, 2)

        self.state_group = QtWidgets.QGroupBox(self.tr("State"))
        self.state_group.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.state_layout = QtWidgets.QGridLayout(self.state_group)
        self.__bubbles = [] # type: list[TaskStateBubble]
        self.bubble_holder = QtWidgets.QWidget()
        self.bubble_layout = QtWidgets.QGridLayout(self.bubble_holder)
        for i in range(self.BUBBLE_ROWS):
            for j in range(self.BUBBLE_COLUMNS):
                bubble = TaskStateBubble()
                self.bubble_layout.addWidget(bubble, i, j)
                self.__bubbles.append(bubble)
        self.previous_button = QtWidgets.QPushButton(self.tr("Previous"))
        self.page_combo_box = QtWidgets.QComboBox()
        self.page_combo_box.addItem(self.tr("No Page"))
        self.next_button = QtWidgets.QPushButton(self.tr("Next"))
        self.previous_button.clicked.connect(lambda: self.page_combo_box.setCurrentIndex(max(self.page_index-1, 0)))
        self.page_combo_box.currentIndexChanged.connect(self.update_page)
        self.next_button.clicked.connect(lambda: self.page_combo_box.setCurrentIndex(min(self.page_index+1, self.n_pages-1)))
        self.state_layout.addWidget(self.bubble_holder, 0, 0, 1, 3)
        self.state_layout.addWidget(self.previous_button, 1, 0)
        self.state_layout.addWidget(self.page_combo_box, 1, 1)
        self.state_layout.addWidget(self.next_button, 1, 2)

        self.main_layout.addWidget(self.control_group, 0, 0)
        self.main_layout.addWidget(self.state_group, 1, 0)

    def setup_processes(self):
        self.__watcher.start()
        n_processes = self.n_workers_input.value()
        for i in range(n_processes):
            process = mp.Process(
                target=excute_tasks,
                args=(self.__task_queue,
                      self.__result_queue,
                      self.__failed_queue,
                      self.__event_queue))
            process.start()
            self.__processes.append(process)

    def kill_processes(self):
        self.__watcher.stop()
        for process in self.__processes:
            process.kill()
            process.join()
        self.__processes.clear()
        self.__task_queue.close()
        self.__failed_queue.close()
        self.__result_queue.close()
        self.__event_queue.close()
        self.__task_queue = mp.Queue()
        self.__result_queue = mp.Queue()
        self.__failed_queue = mp.Queue()
        self.__event_queue = mp.Queue()

    def setup_tasks(self, tasks: typing.List[SSUTask]):
        self.__tasks.clear()
        self.__index_map.clear()
        self.__state_map.clear()
        self.__results.clear()
        self.__failed_tasks.clear()
        for i, task in enumerate(tasks):
            self.__index_map[task.uuid] = i
            self.__state_map[i] = TaskState.NotStarted
            self.__tasks.append(task)
            self.__task_queue.put(task)
        self.n_tasks_display.setText(str(self.n_tasks))
        self.n_remains_display.setText(str(self.n_tasks))
        self.n_failed_tasks_display.setText("0")
        self.n_finished_tasks_display.setText("0")
        self.page_combo_box.clear()
        for i in range(self.n_pages):
            self.page_combo_box.addItem(self.tr("Page {0}").format(i+1))
        self.page_combo_box.setCurrentIndex(0)
        self.n_workers_input.setEnabled(True)
        self.run_button.setEnabled(True)

    def on_run_button_clicked(self):
        self.n_workers_input.setEnabled(False)
        self.run_button.setEnabled(False)
        self.setup_processes()

    def update_page(self):
        if self.page_index < 0:
            return
        offset = self.page_index * self.page_size
        for i in range(self.page_size):
            index = i + offset
            bubble = self.__bubbles[i]
            if index >= self.n_tasks:
                bubble.make_transparent()
                bubble.setToolTip(None)
            else:
                state = self.__state_map[index]
                task = self.__tasks[index]
                bubble.show()
                bubble.change_state(state)
                bubble.setToolTip(task.sample.name)

    def watch(self):
        try:
            while True:
                event = self.__event_queue.get(block=False) # type: typing.Tuple[SSUTask, TaskState]
                task, state = event
                index = self.__index_map[task.uuid]
                self.__state_map[index] = state

                if self.in_page(index):
                    offset = self.page_index*self.page_size
                    bubble = self.__bubbles[index - offset]
                    bubble.change_state(state)
        except Exception:
            pass

        try:
            while True:
                result = self.__result_queue.get(block=False) # type: SSUResult
                index = self.__index_map[result.task.uuid]
                self.__results[index] = result
        except Exception:
            pass

        try:
            while True:
                task = self.__failed_queue.get(block=False) # type: SSUTask
                index = self.__index_map[task.uuid]
                self.__failed_tasks[index] = task
        except Exception:
            pass

        n_remains = 0
        n_failed = 0
        n_finished = 0
        for index, state in self.__state_map.items():
            if state == TaskState.NotStarted or state == TaskState.Processing:
                n_remains += 1
            elif state == TaskState.Failed:
                n_failed += 1
            else:
                n_finished += 1
        self.n_remains_display.setText(str(n_remains))
        self.n_failed_tasks_display.setText(str(n_failed))
        self.n_finished_tasks_display.setText(str(n_finished))

        if n_remains == 0 and n_failed == len(self.__failed_tasks) and n_finished == len(self.__results):
            results = [(index, result) for index, result in self.__results.items()]
            results.sort(key=lambda x: x[0])
            for index, result in results:
                self.result_finished.emit(result)
            self.kill_processes()

    def retranslate(self):
        self.setWindowTitle(self.tr("SSU Multicore Analyzer"))
        self.control_group.setTitle(self.tr("Control"))
        self.n_tasks_label.setText(self.tr("Total Number of Tasks"))
        self.n_workers_label.setText(self.tr("Number of Workers"))
        self.n_remains_label.setText(self.tr("Number of Remaining Tasks"))
        self.n_finished_tasks_label.setText(self.tr("Number of Finished Tasks"))
        self.n_failed_tasks_label.setText(self.tr("Number of Failed Tasks"))
        if self.n_tasks == 0:
            self.n_tasks_display.setText(self.tr("Unknown"))
            self.n_remains_display.setText(self.tr("Unknown"))
            self.n_finished_tasks_display.setText(self.tr("Unknown"))
            self.n_failed_tasks_display.setText(self.tr("Unknown"))
            self.page_combo_box.addItem(self.tr("No Page"))
        else:
            for i in range(self.page_combo_box.count()):
                self.page_combo_box.setItemText(i, self.tr("Page {0}").format(i+1))
        self.run_button.setText(self.tr("Start"))
        self.state_group.setTitle(self.tr("State"))
        self.previous_button.setText(self.tr("Previous"))
        self.next_button.setText(self.tr("Next"))
