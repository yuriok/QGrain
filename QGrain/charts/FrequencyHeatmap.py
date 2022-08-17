__all__ = ["FrequencyHeatmap"]

from typing import *

import numpy as np
from PySide6 import QtGui, QtWidgets
from matplotlib.ticker import FuncFormatter
from numpy import ndarray

from . import BaseChart
from ..models import Sample
from ..statistics import to_microns


class FrequencyHeatmap(BaseChart):
    def __init__(self, parent=None, figsize=(3.3, 4.4)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Frequency Heatmap"))
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
                  ("phi", self.tr("Phi")))
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
            return lambda classes_phi: np.log10(to_microns(classes_phi))
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

    def update_chart(self):
        self._figure.clear()
        self._axes = self._figure.subplots()
        self.show_samples(self._last_samples, append=False)

    def show_samples(self, samples: Sequence[Sample], append=False):
        if len(samples) == 0:
            return
        append = append and len(self._last_samples) != 0
        if not append:
            self._last_samples = []
        self._figure.clear()
        self._axes = self._figure.subplots()
        record_samples = []
        sample_distributions = []
        for sample in self._last_samples:
            record_samples.append(sample)
            sample_distributions.append(sample.distribution)
        for sample in samples:
            record_samples.append(sample)
            sample_distributions.append(sample.distribution)

        Z = np.array(sample_distributions)
        x = self.transfer(record_samples[0].classes_phi)
        y = np.arange(0, len(Z)+1)
        extent = [x[0], x[-1], y[0], y[-1]]
        im = self._axes.imshow(Z[::-1], cmap="coolwarm", aspect="auto", extent=extent, interpolation="gaussian")
        cbar = self._axes.figure.colorbar(im, ax=self._axes)
        cbar.ax.set_ylabel(self.ylabel)
        self._axes.set_xlabel(self.xlabel)
        self._axes.set_ylabel("Sample index")
        self._last_samples = record_samples
        if self.scale == "log-linear":
            self._axes.xaxis.set_major_formatter(FuncFormatter(lambda v, p=None: f"{10 ** v}"))
        self._canvas.draw()

    def retranslate(self):
        self.setWindowTitle(self.tr("Frequency Heatmap"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))
        self.scale_menu.setTitle(self.tr("Scale"))
