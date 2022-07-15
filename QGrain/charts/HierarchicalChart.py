from typing import *

from scipy.cluster.hierarchy import dendrogram, linkage

from . import BaseChart
from ..models import ArtificialDataset, Dataset


class HierarchicalChart(BaseChart):
    def __init__(self, parent=None, figsize=(6.6, 4.4)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Hierarchical Clustering"))
        self.axes = self._figure.subplots()
        self._last_result = None

    def show_result(self, dataset: Union[ArtificialDataset, Dataset], method="ward", metric="euclidean", p=100):
        self._last_result = (dataset, method, metric, p)
        linkage_matrix = linkage(dataset.distributions, method=method, metric=metric)
        self.axes.clear()
        dendrogram(linkage_matrix, no_labels=False, p=p, truncate_mode='lastp',
                   show_contracted=True, ax=self.axes)
        self.axes.set_xlabel("Sample count/index")
        self.axes.set_ylabel("Distance")
        self._figure.tight_layout()
        self._canvas.draw()

    def update_chart(self):
        self._figure.clear()
        self.axes = self._figure.subplots()
        if self._last_result is not None:
            self.show_result(*self._last_result)

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("Hierarchical Clustering"))
