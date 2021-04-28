import typing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout


class DistanceCurveChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Distance Curve Chart"))
        self.figure = plt.figure(figsize=(4, 3))
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()

    def show_distance_series(self, series: typing.Iterable[float], title=""):
        self.axes.clear()
        self.axes.plot(series)
        self.axes.set_xlabel(self.tr("Iteration"))
        self.axes.set_ylabel(self.tr("Distance"))
        self.axes.set_title(title)
        self.figure.tight_layout()
        self.canvas.draw()
