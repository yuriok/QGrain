import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout
from QGrain.models.GrainSizeDataset import GrainSizeDataset
from sklearn.decomposition import PCA


class PCAResultChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("PCA Result Chart"))
        self.figure = plt.figure()
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()

    def show_result(self,
                    dataset: GrainSizeDataset,
                    transformed: np.ndarray,
                    pca: PCA):
        self.axes.clear()
        # X = dataset.X
        components = pca.components_
        # X_hat = transformed @ components
        # MAD = np.abs(X_hat - X)
        n_samples, n_components = transformed.shape
        n_components_, n_classes = components.shape
        sample_indexes = np.linspace(1, n_samples, n_samples)
        assert n_components == n_components_

        for i in range(n_components):
            self.axes.plot(sample_indexes, transformed[:, i], label=f"PC{i+1} ({pca.explained_variance_ratio_[i]:0.2%})")
        self.axes.set_xlabel(self.tr("Sample Index"))
        self.axes.set_ylabel(self.tr("Transformed Value"))
        self.axes.set_title(self.tr("Varations of PCs"))
        if n_components < 10:
            self.axes.legend(loc="upper left")
        self.figure.tight_layout()
        self.canvas.draw()
