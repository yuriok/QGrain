__all__ = ["BoxplotChart"]

from typing import *

import matplotlib.pyplot as plt
from numpy import ndarray

from . import normal_color, BaseChart


class BoxplotChart(BaseChart):
    def __init__(self, parent=None, figsize=(3, 2.5)):
        super().__init__(parent=parent, figsize=figsize)
        self._axes = self._figure.subplots()
        self.setWindowTitle(self.tr("Boxplot Chart"))
        self._last_result = None

    def show_dataset(self, dataset: Sequence[ndarray], xlabels: Sequence[str], ylabel: str, title: str = ""):
        self._axes.clear()
        assert len(dataset) == len(xlabels)
        # "whiskers", "caps", "boxes", "medians", "fliers", "means"
        artists = self._axes.boxplot(dataset, labels=xlabels, patch_artist=True)
        cmap = plt.get_cmap()
        for i, box in enumerate(artists["boxes"]):
            box.set_facecolor(cmap(i))
            box.set_edgecolor(normal_color())
        for whisker in artists["whiskers"]:
            whisker.set_color(normal_color())
        for median in artists["medians"]:
            median.set_color(normal_color())
        for cap in artists["caps"]:
            cap.set_color(normal_color())
        for i, flier in enumerate(artists["fliers"]):
            flier.set_markerfacecolor(cmap(i))
            flier.set_markeredgewidth(0.0)
        self._axes.set_ylabel(ylabel)
        self._axes.set_title(title)
        self._figure.tight_layout()
        self._canvas.draw()
        self._last_result = dataset, xlabels, ylabel, title

    def update_chart(self):
        if self._last_result is not None:
            self._figure.clear()
            self._axes = self._figure.subplots()
            self.show_dataset(*self._last_result)

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("Boxplot Chart"))
