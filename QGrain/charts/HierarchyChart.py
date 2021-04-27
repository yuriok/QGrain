import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout
from scipy.cluster.hierarchy import dendrogram


class HierarchyChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Hierarchy Chart"))
        self.figure = plt.figure(figsize=(8, 4))
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()

    def show_result(self,
                    linkage_matrix: np.ndarray,
                    p=100,
                    **kwargs):
        self.axes.clear()
        res = dendrogram(linkage_matrix,
                         no_labels=False,
                         p=p, truncate_mode='lastp',
                         show_contracted=True,
                         ax=self.axes)
        self.axes.set_title(f"{self.tr('Hierarchy Clustering Chart')} (p={p})")
        self.axes.set_xlabel(self.tr("Sample Count/Index"))
        self.axes.set_ylabel(self.tr("Distance"))
        self.figure.tight_layout()
        self.canvas.draw()
        return res
