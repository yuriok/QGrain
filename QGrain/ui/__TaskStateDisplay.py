# import typing

# from PySide2.QtCore import Qt
# from PySide2.QtGui import (QBrush, QColor, QPainter, QPaintEvent, QPen,
#                            QResizeEvent)
# from PySide2.QtWidgets import (QGridLayout, QLabel, QScrollArea, QSizePolicy,
#                                QWidget)
# from QGrain.algorithms import FittingState
# from QGrain.ssu import SSUTask


# class TaskStateBubble(QWidget):
#     def __init__(self, size=16, border_radius=4):
#         super().__init__()
#         self.setAttribute(Qt.WA_StyledBackground, True)
#         self.__size = size
#         self.__border_radius = border_radius
#         self.initialize_ui()

#     def get_qss_by_state(self, state: FittingState):
#         if state == FittingState.NotStarted:
#             return f"border-radius: {self.__border_radius}px;border:0px;background:#7a7374"
#         elif state == FittingState.Fitting:
#             return f"border-radius: {self.__border_radius}px;border:0px;background:#ff9900"
#         elif state == FittingState.Failed:
#             return f"border-radius: {self.__border_radius}px;border:0px;background:#36292f"
#         elif state == FittingState.Succeeded:
#             return f"border-radius: {self.__border_radius}px;border:0px;background:#8abcd1"
#         else:
#             raise NotImplementedError(state)

#     def get_hint_text_by_state(self, state: FittingState):
#         if state == FittingState.NotStarted:
#             return self.tr("Not Started")
#         elif state == FittingState.Fitting:
#             return self.tr("Fitting")
#         elif state == FittingState.Failed:
#             return self.tr("Failed")
#         elif state == FittingState.Succeeded:
#             return self.tr("Succeeded")
#         else:
#             raise NotImplementedError(state)

#     def change_state(self, state: FittingState):
#         qss = self.get_qss_by_state(state)
#         self.display.setStyleSheet(qss)

#     def initialize_ui(self):
#         self.main_layout = QGridLayout(self)
#         self.display = QLabel()
#         self.display.setFixedSize(self.__size, self.__size)
#         self.main_layout.setContentsMargins(0, 0, 0, 0)
#         self.main_layout.addWidget(self.display)


# class TaskStateBubblePanel(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setAttribute(Qt.WA_StyledBackground, True)
#         self.widgets = []
#         self.initialize_ui()

#     def initialize_ui(self):
#         self.main_layout = QGridLayout(self)
#         demo_names = [f"Test{i+1}" for i in range(100)]
#         self.set_tasks(demo_names)

#     def set_tasks(self, task_names: typing.List[str]):
#         task_count = len(task_names)
#         current_count = len(self.widgets)
#         # prepare the widgets to display
#         if current_count > task_count:
#             for i in range(current_count, task_count, -1):
#                 self.widgets.pop(i-1)
#         elif current_count < task_count:
#             for i in range(current_count, task_count):
#                 w = TaskStateBubble()
#                 w.change_state(FittingState.NotStarted)
#                 self.widgets.append(w)
#         assert len(self.widgets) == task_count

#         # reset the tool tips to show the names of tasks
#         for i, (widget, name) in enumerate(zip(self.widgets, task_names)):
#             widget.setToolTip(f"[{i+1}] {name}")

#     def change_states(self, states: dict):
#         for index, state in states.items():
#             self.widgets[index].change_state(state)

#     def resizeEvent(self, event: QResizeEvent):
#         event.accept()

#         if len(self.widgets) == 0:
#             return
#         max_width = event.size().width()
#         widget_width = self.widgets[0].sizeHint().width()
#         max_column = max_width // (widget_width + self.main_layout.margin()) + 2
#         max_column = max(max_column, 6)

#         if self.main_layout.columnCount() == max_column:
#             return
#         for w in self.widgets:
#             self.main_layout.removeWidget(w)
#         for i, w in enumerate(self.widgets):
#             row = i // max_column
#             col = i % max_column
#             self.main_layout.addWidget(w, row, col)


# class TaskStateDisplay(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.task_list = None
#         self.setAttribute(Qt.WA_StyledBackground, True)
#         self.initialize_ui()

#     def initialize_ui(self):
#         self.main_layout = QGridLayout(self)
#         self.bubble_panel = TaskStateBubblePanel()
#         self.scroll_area = QScrollArea()
#         self.scroll_area.setWidget(self.bubble_panel)
#         self.scroll_area.setWidgetResizable(True)
#         self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#         self.main_layout.addWidget(self.scroll_area)

#         self.setMinimumWidth(600)

#     def set_tasks(self, tasks: typing.List[SSUTask]):
#         self.task_list = tasks
#         self.bubble_panel.set_tasks([task.sample.name for task in tasks])

#     def change_states(self, task_states: dict):
#         states = {}
#         for index, task in enumerate(self.task_list):
#             states[index] = task_states[task.uuid]
#         self.bubble_panel.change_states(states)

