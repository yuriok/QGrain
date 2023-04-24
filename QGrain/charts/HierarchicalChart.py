from typing import *

from numpy import ndarray
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

from . import BaseChart
from ..models import ArtificialDataset, Dataset


class HierarchicalChart(BaseChart):
    def __init__(self, parent=None, figsize=(6.6, 4.4)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Hierarchical Clustering"))
        self._axes = self._figure.subplots()
        self._last_result = None

    def show_dataset(self, dataset: Union[ArtificialDataset, Dataset], method="ward", metric="euclidean", p=100):
        pca = PCA(n_components=0.95)
        transformed = pca.fit_transform(dataset.distributions)
        linkage_matrix = linkage(transformed, method=method, metric=metric)
        self.show_matrix(linkage_matrix, p)

    def show_matrix(self, linkage_matrix: ndarray, p=100):
        self._last_result = (linkage_matrix, p)
        self._axes.clear()
        dendrogram(linkage_matrix, p=p, truncate_mode="lastp", no_labels=False, leaf_font_size=7, ax=self._axes)
        self._axes.set_xlabel(self.tr("Sample index (count)"))
        self._axes.set_ylabel(self.tr("Distance"))
        self._figure.tight_layout()
        self._canvas.draw()

    def update_chart(self):
        self._figure.clear()
        self._axes = self._figure.subplots()
        if self._last_result is not None:
            self.show_matrix(*self._last_result)

    def retranslate(self):
        self.setWindowTitle(self.tr("Hierarchical Clustering"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.configure_subplots_action.setText(self.tr("Configure Subplots"))
        self.save_figure_action.setText(self.tr("Save Figure"))
