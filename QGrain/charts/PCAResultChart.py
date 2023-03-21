from typing import *

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from . import BaseChart
from . import normal_color
from ..models import Dataset, ArtificialDataset


class PCAResultChart(BaseChart):
    def __init__(self, parent=None, size=(6.6, 4.4)):
        super().__init__(parent=parent, figsize=size)
        self.sample_axes = self._figure.add_subplot(2, 2, 1)
        self.shape_axes = self._figure.add_subplot(2, 2, 2)
        self.series_axes = self._figure.add_subplot(2, 1, 2)
        self.setWindowTitle(self.tr("PCA Chart"))
        self._last_dataset: Optional[Dataset] = None

    def show_dataset(self, dataset: Union[ArtificialDataset, Dataset]):
        assert dataset is not None
        self._last_dataset = dataset
        pca = PCA()
        transformed = pca.fit_transform(dataset.distributions)
        self.sample_axes.clear()
        self.shape_axes.clear()
        self.series_axes.clear()
        cmap = plt.get_cmap()
        self.sample_axes.scatter(transformed[:, 0], transformed[:, 1], c=normal_color(), s=2.0, alpha=0.05)
        self.sample_axes.plot(pca.components_[0, :], pca.components_[1, :], color=normal_color(), lw=1.0)
        xi = np.argmax(pca.components_[0, :])
        yi = np.argmax(pca.components_[1, :])
        self.sample_axes.arrow(0, 0, pca.components_[0, xi], pca.components_[1, xi], color=cmap(0), width=0.001)
        self.sample_axes.text(pca.components_[0, xi], pca.components_[1, xi], f"{dataset.classes[xi]: 0.4f}",
                              color=cmap(0), ha='center', va='center')
        self.sample_axes.arrow(0, 0, pca.components_[0, yi], pca.components_[1, yi], color=cmap(1), width=0.001)
        self.sample_axes.text(pca.components_[0, yi], pca.components_[1, yi], f"{dataset.classes[yi]: 0.4f}",
                              color=cmap(1), ha='center', va='center')
        self.sample_axes.set_xlabel(r"$\rm PC_1$")
        self.sample_axes.set_ylabel(r"$\rm PC_2$")
        cumulative_ratio = 0.0
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            cumulative_ratio += ratio
            self.shape_axes.plot(dataset.classes, pca.components_[i], color=cmap(i), label=r"$\rm PC_{0}$".format(i+1))
            self.series_axes.plot(transformed[:, i], color=cmap(i),
                                  label=r"$\rm PC_{0}$".format(i+1) + f" ({pca.explained_variance_ratio_[i]:0.2%})",
                                  lw=1.0, alpha=0.8)
            if i > 0 and cumulative_ratio > 0.95:
                break
        self.shape_axes.set_xscale("log")
        self.shape_axes.set_xlabel(self.tr("Grain size ({0})").format(r"$\rm \mu m$"))
        self.shape_axes.set_ylabel(self.tr("Transformed value"))
        self.shape_axes.legend(loc="upper left", prop={"size": 6})
        self.series_axes.set_xlabel(self.tr("Sample index"))
        self.series_axes.set_ylabel(self.tr("Transformed value"))
        self.series_axes.legend(loc="upper left", prop={"size": 6})
        self._figure.tight_layout()
        self._canvas.draw()

    def update_chart(self):
        self._figure.clear()
        self.sample_axes = self._figure.add_subplot(2, 2, 1)
        self.shape_axes = self._figure.add_subplot(2, 2, 2)
        self.series_axes = self._figure.add_subplot(2, 1, 2)
        if self._last_dataset is not None:
            self.show_dataset(self._last_dataset)

    def retranslate(self):
        self.setWindowTitle(self.tr("PCA Chart"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.configure_subplots_action.setText(self.tr("Configure Subplots"))
        self.save_figure_action.setText(self.tr("Save Figure"))
