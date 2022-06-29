__all__ = ["BaseChart"]

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6 import QtCore, QtGui, QtWidgets

from .config_matplotlib import setup_matplotlib


class BaseChart(QtWidgets.QWidget):
    def __init__(self, parent=None, figsize=(4, 3)):
        super().__init__(parent=parent)
        self.figure = plt.figure(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.canvas, 0, 0)

        self.menu = QtWidgets.QMenu(self.canvas)
        self.menu.setShortcutAutoRepeat(True)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)
        self.edit_figure_action = self.menu.addAction(self.tr("Edit Figure"))
        self.edit_figure_action.triggered.connect(lambda: self.toolbar.edit_parameters())
        self.save_figure_action = self.menu.addAction(self.tr("Save Figure"))
        self.save_figure_action.triggered.connect(lambda: self.toolbar.save_figure())
        self.normal_msg = QtWidgets.QMessageBox(parent=self)

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

    def show_menu(self, pos: QtCore.QPoint):
        self.menu.popup(QtGui.QCursor.pos())

    def update_chart(self):
        pass

    def retranslate(self):
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.StyleChange:
            setup_matplotlib()
            self.figure.clear()
            self.main_layout.removeWidget(self.canvas)
            self.canvas.setVisible(False)
            self.figure = plt.figure(figsize=self.figure.get_size_inches())
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.main_layout.addWidget(self.canvas, 0, 0)
            self.update_chart()

        elif event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()
