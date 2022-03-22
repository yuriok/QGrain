import typing

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtWidgets
from sklearn.decomposition import PCA

from ..model import GrainSizeDataset
from .BaseChart import BaseChart
from .config_matplotlib import highlight_color, normal_color


class PCAResultChart(BaseChart):
    def __init__(self, parent=None, figsize=(6, 5)):
        super().__init__(parent=parent, figsize=figsize)
        self.sample_axes = self.figure.add_subplot(2, 2, 1)
        self.shape_axes = self.figure.add_subplot(2, 2, 2)
        self.series_axes = self.figure.add_subplot(2, 1, 2)
        self.setWindowTitle(self.tr("PCA Chart"))
        self.__last_dataset = None # type: GrainSizeDataset

    def show_dataset(self, dataset: GrainSizeDataset):
        assert dataset is not None
        assert dataset.has_sample
        self.__last_dataset = dataset

        X = dataset.distribution_matrix
        pca = PCA()
        transformed = pca.fit_transform(X)
        # labels = [f"{v:0.4f}" for v in dataset.classes_μm]
        n_samples, n_features = X.shape
        cmap = plt.get_cmap()
        self.sample_axes.clear()
        self.shape_axes.clear()
        self.series_axes.clear()

        self.sample_axes.scatter(transformed[:, 0], transformed[:, 1], c=normal_color(), s=2.0, alpha=0.05)
        self.sample_axes.plot(pca.components_[0, :], pca.components_[1, :], color=normal_color(), lw=1.0, alpha=1.0)
        xi = np.argmax(pca.components_[0, :])
        yi = np.argmax(pca.components_[1, :])
        self.sample_axes.arrow(0, 0, pca.components_[0, xi], pca.components_[1, xi], color=cmap(0), width=0.001, alpha=1.0)
        self.sample_axes.text(pca.components_[0, xi], pca.components_[1, xi], f"{dataset.classes_μm[xi]: 0.4f}", color=cmap(0), ha='center', va='center')
        self.sample_axes.arrow(0, 0, pca.components_[0, yi], pca.components_[1, yi], color=cmap(1), width=0.001, alpha=1.0)
        self.sample_axes.text(pca.components_[0, yi], pca.components_[1, yi], f"{dataset.classes_μm[yi]: 0.4f}", color=cmap(1), ha='center', va='center')
        self.sample_axes.set_xlabel("PC1")
        self.sample_axes.set_ylabel("PC2")

        self.shape_axes.plot(dataset.classes_μm, pca.components_[0], color=cmap(0), label="PC1")
        self.shape_axes.plot(dataset.classes_μm, pca.components_[1], color=cmap(1), label="PC2")
        self.shape_axes.set_xscale("log")
        # plt.xticks([1e-1, 1e0, 1e1, 1e2, 1e3], ["0.1", "1", "10", "100", "1000"])
        self.shape_axes.set_xlabel("Grain size [μm]")
        self.shape_axes.set_ylabel("Transformed value")
        self.shape_axes.legend(loc="upper left")

        for i in range(2):
            self.series_axes.plot(transformed[:, i], color=cmap(i), label=f"PC{i+1} ({pca.explained_variance_ratio_[i]:0.2%})", lw=1.0, alpha=0.8)
        self.series_axes.set_xlabel("Sample index")
        self.series_axes.set_ylabel("Transformed value")
        self.series_axes.legend(loc="upper left")
        self.figure.tight_layout()
        self.canvas.draw()

    def update_chart(self):
        self.figure.clear()
        self.sample_axes = self.figure.add_subplot(2, 2, 1)
        self.shape_axes = self.figure.add_subplot(2, 2, 2)
        self.series_axes = self.figure.add_subplot(2, 1, 2)
        if self.__last_dataset is not None:
            self.show_dataset(self.__last_dataset)

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("PCA Chart"))
