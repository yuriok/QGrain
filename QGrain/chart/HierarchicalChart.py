import typing

import numpy as np
from PySide6 import QtCore, QtWidgets
from scipy.cluster.hierarchy import dendrogram

from .BaseChart import BaseChart


class HierarchicalChart(BaseChart):
    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Hierarchical Clustering"))
        self.axes = self.figure.subplots()
        self.__linkage_matrix = None
        self.__p = None

    def show_result(self,
                    linkage_matrix: np.ndarray,
                    p=100):
        self.axes.clear()
        dendrogram(linkage_matrix,
                   no_labels=False,
                   p=p, truncate_mode='lastp',
                   show_contracted=True,
                   ax=self.axes)
        self.axes.set_xlabel("Sample count/index")
        self.axes.set_ylabel("Distance")
        self.figure.tight_layout()
        self.canvas.draw()

    def update_chart(self):
        self.figure.clear()
        self.axes = self.figure.subplots()
        if self.__linkage_matrix is not None:
            self.show_result(self.__linkage_matrix, self.__p)
