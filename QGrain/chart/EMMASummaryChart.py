import typing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDialog, QGridLayout, QLabel

from ..emma import EMMAResult


class EMMASummaryChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("EMMA Summary Chart"))
        self.figure = plt.figure(figsize=(4, 3))
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)

        self.supported_distances = ("1-norm", "2-norm", "3-norm", "4-norm", "MSE", "log10MSE", "cosine", "angular")
        self.distance_label = QLabel(self.tr("Distance"))
        self.distance_combo_box = QComboBox()
        self.distance_combo_box.addItems(self.supported_distances)
        self.distance_combo_box.setCurrentText("log10MSE")
        self.distance_combo_box.currentIndexChanged.connect(self.update_chart)
        self.main_layout.addWidget(self.distance_label, 2, 0)
        self.main_layout.addWidget(self.distance_combo_box, 2, 1)

        if not toolbar:
            self.toolbar.hide()

        self.results = []

    @property
    def distance(self) -> str:
        return self.distance_combo_box.currentText()

    def update_chart(self):
        self.show_distances(self.results)

    def show_distances(self, results: typing.List[EMMAResult], title=""):
        self.results = results

        n_members_list = [result.n_members for result in results]
        distances = [result.get_distance(self.distance) for result in results]

        self.axes.clear()
        self.axes.plot(n_members_list, distances, c="black", linewidth=2.5, marker=".", ms=8, mfc="black", mew=0.0)
        self.axes.set_xlabel(self.tr("$N_{members}$"))
        self.axes.set_ylabel(self.tr("Distance"))
        self.axes.set_title(title)
        self.figure.tight_layout()
        self.canvas.draw()
