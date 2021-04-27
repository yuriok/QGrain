import typing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout


class BoxplotChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Boxplot Chart"))
        self.figure = plt.figure()
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()

    def show_dataset(self, dataset: typing.Iterable[typing.Iterable[float]],
                     xlabels, ylabel,
                     title=""):
        self.axes.clear()
        self.axes.boxplot(dataset, labels=xlabels)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)
        self.figure.tight_layout()
        self.canvas.draw()
