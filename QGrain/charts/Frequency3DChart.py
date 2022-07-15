__all__ = ["Frequency3DChart"]

from typing import *

import numpy as np
from PySide6 import QtGui, QtWidgets
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray

from . import BaseChart
from ..models import Sample
from ..statistics import to_microns


class Frequency3DChart(BaseChart):
    def __init__(self, parent=None, figsize=(6, 4)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Frequency 3D Chart"))
        self._axes = Axes3D(self._figure, auto_add_to_figure=False)
        self._figure.add_axes(self._axes)
        self.scale_menu: QtWidgets.QMenu = QtWidgets.QMenu(self.tr("Scale"))
        self.menu.insertMenu(self.edit_figure_action, self.scale_menu)
        self.scale_group = QtGui.QActionGroup(self.scale_menu)
        self.scale_group.setExclusive(True)
        self.scale_actions: List[QtGui.QAction] = []
        for key, name in self.supported_scales:
            scale_action: QtGui.QAction = self.scale_group.addAction(name)
            scale_action.setCheckable(True)
            scale_action.triggered.connect(self.update_chart)
            self.scale_menu.addAction(scale_action)
            self.scale_actions.append(scale_action)
        self.scale_actions[2].setChecked(True)
        self._last_samples = []

    @property
    def supported_scales(self) -> Tuple[Tuple[str, str]]:
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
        self._axes = Axes3D(self._figure, auto_add_to_figure=False)
        self._figure.add_axes(self._axes)
        self.show_samples(self._last_samples, append=False)

    def show_samples(self, samples: Sequence[Sample], append=False):
        if len(samples) == 0:
            return
        append = append and len(self._last_samples) != 0
        if not append:
            self._last_samples = []
        self._axes.clear()
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
        y = np.linspace(1, len(Z), len(Z))
        X, Y = np.meshgrid(x, y)
        self._axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="binary")
        self._axes.set_xlim(x[0], x[-1])
        self._axes.set_xlabel(self.xlabel)
        self._axes.set_ylabel(self.tr("Sample index"))
        self._axes.set_zlabel(self.ylabel)
        self._last_samples = record_samples
        if self.scale == "linear":
            self._axes.view_init(elev=15.0, azim=45)
        else:
            self._axes.view_init(elev=45.0, azim=-120)
        self._canvas.draw()

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("Frequency 3D Chart"))
        self.scale_menu.setTitle(self.tr("Scale"))
