import typing

import matplotlib.pyplot as plt
import numpy as np

from .BaseChart import BaseChart
from .config_matplotlib import highlight_color, normal_color


class BoxplotChart(BaseChart):
    def __init__(self, parent=None, figsize=(4, 3)):
        super().__init__(parent=parent, figsize=figsize)
        self.axes = self.figure.subplots()
        self.setWindowTitle(self.tr("Boxplot Chart"))
        self.__last_result = None

    def show_dataset(self, dataset: typing.List[np.ndarray],
                     xlabels: typing.List[str], ylabel: str,
                     title: str = ""):
        self.axes.clear()
        assert len(dataset) == len(xlabels)
        # "whiskers", "caps", "boxes", "medians", "fliers", "means"
        artists = self.axes.boxplot(dataset, labels=xlabels, patch_artist=True)
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
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)
        self.figure.tight_layout()
        self.canvas.draw()
        self.__last_result = dataset, xlabels, ylabel, title

    def update_chart(self):
        if self.__last_result is not None:
            self.figure.clear()
            self.axes = self.figure.subplots()
            self.show_dataset(*self.__last_result)

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("Boxplot Chart"))
