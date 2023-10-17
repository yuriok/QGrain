__all__ = ["CumulativeChart"]

from typing import *

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtGui, QtWidgets
from numpy import ndarray
from scipy.stats import norm

from . import BaseChart
from ..models import Sample
from ..statistics import to_microns, to_cumulative


class CumulativeChart(BaseChart):
    def __init__(self, parent=None, figsize=(3, 2.5)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Cumulative Frequency Chart"))
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
        self._last_samples = []

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
    def transfer(self) -> Callable[[Union[int, float, ndarray]], Union[int, float, ndarray]]:
        if self.scale == "log-linear":
            return lambda classes_phi: to_microns(classes_phi)
        elif self.scale == "log":
            return lambda classes_phi: np.log(to_microns(classes_phi))
        elif self.scale == "phi":
            return lambda classes_phi: classes_phi
        elif self.scale == "linear":
            return lambda classes_phi: to_microns(classes_phi)

    @property
    def x_label(self) -> str:
        if self.scale == "log-linear":
            return self.tr("Grain size ({0})").format(r"$\rm \mu m$")
        elif self.scale == "log":
            return self.tr("Ln(grain size) ({0})").format(r"$\rm \mu m$")
        elif self.scale == "phi":
            return self.tr("Grain size ({0})").format(r"$\rm \phi$")
        elif self.scale == "linear":
            return self.tr("Grain size ({0})").format(r"$\rm \mu m$")

    @property
    def y_label(self) -> str:
        return self.tr("Frequency ({0})").format(r"$\%$")

    @property
    def xlog(self) -> bool:
        if self.scale == "log-linear":
            return True
        else:
            return False

    def update_chart(self):
        self._figure.clear()
        self._axes = self._figure.subplots()
        self.show_samples(self._last_samples, append=False)

    def show_samples(self, samples: Sequence[Sample], append=False, title=""):
        if len(samples) == 0:
            return
        append = append and len(self._last_samples) != 0
        if not append:
            self._axes.clear()
            self._last_samples = []
            if self.xlog:
                self._axes.set_xscale("log")
            x = self.transfer(samples[0].classes_phi)
            y_ticks = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999]
            self._axes.set_title(title)
            self._axes.set_xlabel(self.x_label)
            self._axes.set_ylabel(self.y_label)
            self._axes.set_yticks(norm.ppf(y_ticks), [f"{y * 100}" for y in y_ticks])
            self._axes.set_yticks([], minor=True)
            self._axes.set_xlim(x[0], x[-1])
            self._axes.set_ylim(norm.ppf(0.0001), norm.ppf(0.9999))
        for i, sample in enumerate(samples):
            self._last_samples.append(sample)
            x = self.transfer(samples[0].classes_phi)
            cumulative_frequency = to_cumulative(sample.distribution)
            c = plt.get_cmap()(i % 10)
            self._axes.plot(x, norm.ppf(cumulative_frequency), c=c, marker=".", mfc=c, mec=c, label=sample.name)
        self._figure.tight_layout()
        self._canvas.draw()

    def retranslate(self):
        self.setWindowTitle(self.tr("Cumulative Frequency Chart"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.configure_subplots_action.setText(self.tr("Configure Subplots"))
        self.save_figure_action.setText(self.tr("Save Figure"))
        self.scale_menu.setTitle(self.tr("Scale"))
