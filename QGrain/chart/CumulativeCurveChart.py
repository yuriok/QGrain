import typing

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..model import GrainSizeSample
from ..statistics import convert_φ_to_μm, get_cumulative_frequency
from .BaseChart import BaseChart
from .config_matplotlib import normal_color


class CumulativeCurveChart(BaseChart):
    def __init__(self, parent=None, figsize=(4, 3)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Cumulative Curve Chart"))
        self.axes = self.figure.subplots()

        self.scale_menu = QtWidgets.QMenu(self.tr("Scale")) # type: QtWidgets.QMenu
        self.menu.insertMenu(self.edit_figure_action, self.scale_menu)
        self.scale_group = QtGui.QActionGroup(self.scale_menu)
        self.scale_group.setExclusive(True)
        self.scale_actions = [] # type: list[QtGui.QAction]
        for key, name in self.supported_scales:
            scale_action = self.scale_group.addAction(name) # type: QtGui.QAction
            scale_action.setCheckable(True)
            scale_action.triggered.connect(self.update_chart)
            self.scale_menu.addAction(scale_action)
            self.scale_actions.append(scale_action)
        self.scale_actions[0].setChecked(True)

        self.__last_samples = []

    @property
    def supported_scales(self) -> typing.List[typing.Tuple[str, str]]:
        scales = [("log-linear", self.tr("Log-linear")),
                  ("log", self.tr("Log")),
                  ("phi", self.tr("Phi")),
                  ("linear", self.tr("Linear"))]
        return scales

    @property
    def scale(self) -> str:
        for i, scale_action in enumerate(self.scale_actions):
            if scale_action.isChecked():
                key, name = self.supported_scales[i]
                return key

    @property
    def transfer(self) -> typing.Callable:
        if self.scale == "log-linear":
            return lambda classes_φ: convert_φ_to_μm(classes_φ)
        elif self.scale == "log":
            return lambda classes_φ: np.log(convert_φ_to_μm(classes_φ))
        elif self.scale == "phi":
            return lambda classes_φ: classes_φ
        elif self.scale == "linear":
            return lambda classes_φ: convert_φ_to_μm(classes_φ)

    @property
    def xlabel(self) -> str:
        if self.scale == "log-linear":
            return "Grain size [μm]"
        elif self.scale == "log":
            return "Ln(grain size in μm)"
        elif self.scale == "phi":
            return "Grain size [φ]"
        elif self.scale == "linear":
            return "Grain size [μm]"

    @property
    def ylabel(self) -> str:
        return "Frequency"

    @property
    def xlog(self) -> bool:
        if self.scale == "log-linear":
            return True
        else:
            return False

    def update_chart(self):
        self.figure.clear()
        self.axes = self.figure.subplots()
        self.show_samples(self.__last_samples, append=False)

    def show_samples(self, samples: typing.Iterable[GrainSizeSample], append=False):
        if len(samples) == 0:
            return
        append = append and len(self.__last_samples) != 0
        if not append:
            self.axes.clear()
            self.__last_samples = []
        for i, sample in enumerate(samples):
            self.__last_samples.append(sample)
            if i == 0:
                x = self.transfer(sample.classes_φ)
                if not append:
                    if self.xlog:
                        self.axes.set_xscale("log")
                    self.axes.set_title("Cumulative curves of samples")
                    self.axes.set_xlabel(self.xlabel)
                    self.axes.set_ylabel(self.ylabel)
                    self.axes.set_xlim(x[0], x[-1])
                    self.axes.set_ylim(0.0, 1.0)
            cumulative_frequency = get_cumulative_frequency(sample.distribution)
            self.axes.plot(x, cumulative_frequency, c=normal_color())
        self.figure.tight_layout()
        self.canvas.draw()

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("Cumulative Curve Chart"))
        self.scale_menu.setTitle(self.tr("Scale"))
