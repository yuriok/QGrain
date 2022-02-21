__all__ = ["LogDialog", "StatusBarLogHandler", "GUILogHandler"]

import logging
import os
from queue import Queue

from PySide6 import QtCore, QtGui, QtWidgets

SUCCESS_COLOR = "#55bb8a"
WARNING_COLOR = "#fbda41"
ERROR_COLOR = "#c04851"

class LogDialog(QtWidgets.QDialog):
    MAX_SIZE = 200
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=QtCore.Qt.Window)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Runtime Log"))
        self.setMinimumSize(400, 400)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(self.MAX_SIZE)
        self.main_layout.addWidget(self.text, 0, 0)
        self.record_queue = Queue(maxsize=self.MAX_SIZE)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("Runtime Log"))

    def get_html_message(self, record: logging.LogRecord, message: str):
        format_str = "<font color='{0}'>{1}<font/>"
        if record.levelno < logging.INFO:
            return message
        elif record.levelno < logging.WARNING:
            return format_str.format(SUCCESS_COLOR, message)
        elif record.levelno < logging.ERROR:
            return format_str.format(WARNING_COLOR, message)
        else:
            return format_str.format(ERROR_COLOR, message)

    def add_record(self, record: logging.LogRecord, message: str):
        if self.record_queue.full():
            self.record_queue.get(False)
        self.record_queue.put((record, message), False)
        html_message = self.get_html_message(record, message)
        self.text.appendHtml(html_message)

class StatusBarLogHandler(logging.Handler):
    def __init__(self, status_bar: QtWidgets.QStatusBar, level=logging.WARNING):
        super().__init__(level=level)
        self.status_bar = status_bar
        self.__mutex = QtCore.QMutex()

    def set_color(self, level: int):
        if level < logging.INFO:
            self.status_bar.setStyleSheet("")
        elif level < logging.WARNING:
            self.status_bar.setStyleSheet("QStatusBar {color: #55bb8a}")
        elif level < logging.ERROR:
            self.status_bar.setStyleSheet("QStatusBar {color: #fbda41}")
        else:
            self.status_bar.setStyleSheet("QStatusBar {color: #c04851}")

    def emit(self, record: logging.LogRecord):
        self.__mutex.lock()
        if record.levelno < self.level:
            return
        self.set_color(record.levelno)
        message = self.format(record)
        self.status_bar.showMessage(message)
        self.__mutex.unlock()

class GUILogHandler(logging.Handler):
    def __init__(self, log_panel: LogDialog, level=logging.WARNING):
        super().__init__(level=level)
        self.log_panel = log_panel
        self.__mutex = QtCore.QMutex()

    def emit(self, record: logging.LogRecord):
        self.__mutex.lock()
        if record.levelno < self.level:
            return
        message = self.format(record)
        self.log_panel.add_record(record, message)
        self.__mutex.unlock()
