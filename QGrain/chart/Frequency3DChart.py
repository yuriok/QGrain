import typing

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..model import GrainSizeSample
from ..statistics import convert_φ_to_μm, get_cumulative_frequency
from .BaseChart import BaseChart
from .config_matplotlib import normal_color
from mpl_toolkits.mplot3d import Axes3D

class Frequency3DChart(BaseChart):
    def __init__(self, parent=None, figsize=(6, 4)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Frequency 3D Chart"))
        self.axes = Axes3D(self.figure, auto_add_to_figure=False)
        self.figure.add_axes(self.axes)

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
        self.scale_actions[2].setChecked(True)

        self.last_samples = []
        self.last_max_frequency = 0.0


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

    def update_chart(self):
        self.figure.clear()
        self.axes = Axes3D(self.figure, auto_add_to_figure=False)
        self.figure.add_axes(self.axes)
        self.show_samples(self.last_samples, append=False)

    def show_samples(self, samples: typing.Iterable[GrainSizeSample], append=False):
        if len(samples) == 0:
            return
        append = append and len(self.last_samples) != 0
        if not append:
            self.last_samples = []
        self.axes.clear()
        record_samples = []
        sample_distributions = []
        for sample in self.last_samples:
            record_samples.append(sample)
            sample_distributions.append(sample.distribution)
        for sample in samples:
            record_samples.append(sample)
            sample_distributions.append(sample.distribution)
        Z = np.array(sample_distributions)
        x = self.transfer(record_samples[0].classes_φ)
        y = np.linspace(1, len(Z), len(Z))
        X, Y = np.meshgrid(x, y)
        self.axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="binary")
        self.axes.set_xlim(x[0], x[-1])
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.tr("Sample index"))
        self.axes.set_zlabel(self.ylabel)
        self.last_samples = record_samples
        if self.scale == "linear":
            self.axes.view_init(elev=15.0, azim=45)
        else:
            self.axes.view_init(elev=45.0, azim=-120)
        self.canvas.draw()

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("Frequency 3D Chart"))
        self.scale_menu.setTitle(self.tr("Scale"))
