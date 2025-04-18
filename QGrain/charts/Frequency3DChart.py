__all__ = ["Frequency3DChart"]

from typing import *

import numpy as np
from PySide6 import QtGui, QtWidgets
from matplotlib.ticker import FuncFormatter
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
            return lambda classes_phi: np.log10(to_microns(classes_phi))
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
        return self.tr("Sample index")

    @property
    def z_label(self) -> str:
        return self.tr("Frequency ({0})").format(r"$\%$")

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
        Z = np.array(sample_distributions)*100
        x = self.transfer(record_samples[0].classes_phi)
        y = np.linspace(1, len(Z), len(Z))
        X, Y = np.meshgrid(x, y)
        self._axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="binary")
        self._axes.set_xlim(x[0], x[-1])
        self._axes.set_xlabel(self.x_label)
        self._axes.set_ylabel(self.y_label)
        self._axes.set_zlabel(self.z_label)
        self._last_samples = record_samples
        if self.scale == "log-linear":
            self._axes.xaxis.set_major_formatter(FuncFormatter(lambda v, p=None: f"{10**v}"))
        self._axes.view_init(elev=20.0, azim=-120)
        self._canvas.draw()

    def retranslate(self):
        self.setWindowTitle(self.tr("Frequency 3D Chart"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.configure_subplots_action.setText(self.tr("Configure Subplots"))
        self.save_figure_action.setText(self.tr("Save Figure"))
        self.scale_menu.setTitle(self.tr("Scale"))
