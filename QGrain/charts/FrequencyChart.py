__all__ = ["FrequencyChart"]

from typing import *

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtGui, QtWidgets
from numpy import ndarray

from . import BaseChart
from ..models import Sample
from ..statistics import to_microns


class FrequencyChart(BaseChart):
    def __init__(self, parent=None, figsize=(3, 2.5)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Frequency Chart"))
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
            self._axes.set_title(title)
            self._axes.set_xlabel(self.xlabel)
            self._axes.set_ylabel(self.ylabel)
            self._axes.set_xlim(x[0], x[-1])
            distributions = np.array([sample.distribution for sample in samples])
            self._axes.set_ylim(0.0, round(np.max(distributions) * 1.2, 2))
        for i, sample in enumerate(samples):
            self._last_samples.append(sample)
            x = self.transfer(sample.classes_phi)
            c = plt.get_cmap()(i % 10)
            self._axes.plot(x, sample.distribution, c=c, marker=".", mfc=c, mec=c, label=sample.name)
        self._figure.tight_layout()
        self._canvas.draw()

    def retranslate(self):
        self.setWindowTitle(self.tr("Frequency Chart"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))
        self.scale_menu.setTitle(self.tr("Scale"))
