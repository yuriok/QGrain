__all__ = ["FrequencyGroupChart"]

from typing import *

import numpy as np
from PySide6 import QtGui, QtWidgets, QtCore
from numpy import ndarray

from . import BaseChart, normal_color
from ..statistics import to_microns


class FrequencyGroupChart(BaseChart):
    def __init__(self, parent=None, figsize=(3, 2.5)):
        super().__init__(parent=parent, figsize=figsize)
        self._axes = self._figure.subplots()
        self.scale_menu = QtWidgets.QMenu(self.tr("Scale"))
        self.menu.insertMenu(self.edit_figure_action, self.scale_menu)
        self.scale_group = QtGui.QActionGroup(self.scale_menu)
        self.scale_group.setExclusive(True)
        self.scale_actions: List[QtGui.QAction] = []
        for key, name in self.supported_scales:
            scale_action = self.scale_group.addAction(name)
            scale_action.setCheckable(True)
            scale_action.triggered.connect(self.update_chart)
            self.scale_menu.addAction(scale_action)
            self.scale_actions.append(scale_action)
        self.scale_actions[0].setChecked(True)
        self._last_group = None
        self.setWindowFlag(QtCore.Qt.SplashScreen)

    @property
    def supported_scales(self) -> Sequence[Tuple[str, str]]:
        scales = (("log-linear", self.tr("Log-linear")),
                  ("log", self.tr("Log")),
                  ("phi", self.tr("Phi")),
                  ("linear", self.tr("Linear")))
        return scales

    @property
    def scale(self) -> str:
        for i, scale_action in enumerate(self.scale_actions):
            if scale_action.isChecked():
                key, name = self.supported_scales[i]
                return key

    @property
    def transfer(self) -> Callable[[Union[float, ndarray]], Union[float, ndarray]]:
        if self.scale == "log-linear":
            return lambda classes_phi: to_microns(classes_phi)
        elif self.scale == "log":
            return lambda classes_phi: np.log(to_microns(classes_phi))
        elif self.scale == "phi":
            return lambda classes_phi: classes_phi
        elif self.scale == "linear":
            return lambda classes_phi: to_microns(classes_phi)

    @property
    def xlabel(self) -> str:
        if self.scale == "log-linear":
            return "Grain size (microns)"
        elif self.scale == "log":
            return "Ln(grain size in microns)"
        elif self.scale == "phi":
            return "Grain size (phi)"
        elif self.scale == "linear":
            return "Grain size (microns)"

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
        self._figure.clear()
        self._axes = self._figure.subplots()
        if self._last_group is not None:
            self.show_frequency_grop(*self._last_group)

    def show_frequency_grop(self, classes_phi: np.ndarray, distributions: np.ndarray, title=""):
        if len(distributions) == 0:
            return
        self._axes.clear()
        self._last_group = (classes_phi, distributions, title)
        if self.xlog:
            self._axes.set_xscale("log")
        x = self.transfer(classes_phi)

        def summarize(distributions: np.ndarray, q=0.05):
            median = np.median(distributions, axis=0)
            upper = np.quantile(distributions, q=1 - q, axis=0)
            lower = np.quantile(distributions, q=q, axis=0)
            return median, lower, upper

        median, upper, lower = summarize(distributions, q=0.01)
        self._axes.fill_between(x, upper, lower, color=normal_color(), linewidth=0.0, alpha=0.2)
        median, upper, lower = summarize(distributions, q=0.05)
        self._axes.fill_between(x, upper, lower, color=normal_color(), linewidth=0.0, alpha=0.4)
        self._axes.plot(x, median, color=normal_color(), linewidth=1.0, linestyle="--")

        self._axes.set_title(title)
        self._axes.set_xlabel(self.xlabel)
        self._axes.set_ylabel(self.ylabel)
        self._axes.set_xlim(x[0], x[-1])
        self._axes.set_ylim(0.0, None)
        self._figure.tight_layout()
        self._canvas.draw()

    def retranslate(self):
        self.setWindowTitle(self.tr("Frequency Chart"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))
        self.scale_menu.setTitle(self.tr("Scale"))
