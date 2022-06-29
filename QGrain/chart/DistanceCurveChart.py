import typing

import matplotlib.pyplot as plt

from .BaseChart import BaseChart


class DistanceCurveChart(BaseChart):
    def __init__(self, parent=None, figsize=(4, 3)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Distance Series"))
        self.axes = self.figure.subplots()
        self.__last_result = None

    def show_distance_series(self, series: typing.Iterable[float], ylabel: str, title: str = ""):
        self.axes.clear()
        self.axes.plot(series, label="series")
        self.axes.set_xlabel("Iteration index")
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)
        self.figure.tight_layout()
        self.canvas.draw()
        self.__last_result = series, ylabel, title

    def update_chart(self):
        if self.__last_result is not None:
            self.figure.clear()
            self.axes = self.figure.subplots()
            self.show_distance_series(*self.__last_result)

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("Distance Series"))
