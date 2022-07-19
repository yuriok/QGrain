__all__ = ["DistributionChart"]

from typing import *

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.animation import FuncAnimation
from numpy import ndarray

from . import BaseChart
from . import normal_color
from ..metrics import loss_numpy
from ..models import SSUResult, ArtificialSample
from ..statistics import to_microns, mode


class DistributionChart(BaseChart):
    def __init__(self, parent=None, size=(3, 2.5)):
        super().__init__(parent=parent, figsize=size)
        self._axes: plt.Axes = self._figure.subplots()
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
        self.show_mode_lines_action = QtGui.QAction(self.tr("Show Mode Lines"))
        self.show_mode_lines_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.show_mode_lines_action)
        self.show_mode_lines_action.setCheckable(True)
        self.show_mode_lines_action.setChecked(False)
        self.show_legend_action = QtGui.QAction(self.tr("Show Legend"))
        self.show_legend_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.show_legend_action)
        self.show_legend_action.setCheckable(True)
        self.show_legend_action.setChecked(False)
        self.animated_action = QtGui.QAction(self.tr("Animated"))
        self.animated_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.animated_action)
        self.animated_action.setCheckable(True)
        self.animated_action.setChecked(False)
        self.interval_menu = QtWidgets.QMenu(self.tr("Animation Interval"))
        self.menu.insertMenu(self.edit_figure_action, self.interval_menu)
        self.interval_group = QtGui.QActionGroup(self.interval_menu)
        self.interval_group.setExclusive(True)
        self.interval_actions: List[QtGui.QAction] = []
        for interval, name in self.supported_intervals:
            interval_action = self.interval_group.addAction(name)
            interval_action.setCheckable(True)
            interval_action.triggered.connect(self.update_chart)
            self.interval_menu.addAction(interval_action)
            self.interval_actions.append(interval_action)
        self.interval_actions[3].setChecked(True)
        self.repeat_animation_action = QtGui.QAction(self.tr("Repeat Animation"))
        self.repeat_animation_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.repeat_animation_action)
        self.repeat_animation_action.setCheckable(True)
        self.repeat_animation_action.setChecked(False)
        self.save_animation_action = QtGui.QAction(self.tr("Save Animation"))
        self.menu.addAction(self.save_animation_action)
        self.save_animation_action.triggered.connect(lambda: self.save_animation())
        self._last_result: Union[ArtificialSample, SSUResult, None] = None

    @property
    def supported_scales(self) -> Sequence[Tuple[str, str]]:
        scales = (("log-linear", self.tr("Log-linear")),
                  ("log", self.tr("Log")),
                  ("phi", self.tr("Phi")),
                  ("linear", self.tr("Linear")))
        return scales

    @property
    def supported_intervals(self) -> Sequence[Tuple[int, str]]:
        intervals = ((5, self.tr("5 Milliseconds")),
                     (10, self.tr("10 Milliseconds")),
                     (20, self.tr("20 Milliseconds")),
                     (30, self.tr("30 Milliseconds")),
                     (60, self.tr("60 Milliseconds")))
        return intervals

    @property
    def scale(self) -> str:
        for i, scale_action in enumerate(self.scale_actions):
            if scale_action.isChecked():
                key, name = self.supported_scales[i]
                return key

    @property
    def show_mode(self) -> bool:
        return self.show_mode_lines_action.isChecked()

    @property
    def show_legend(self) -> bool:
        return self.show_legend_action.isChecked()

    @property
    def animated(self) -> bool:
        return self.animated_action.isChecked()

    @property
    def animation_interval(self) -> int:
        for i, interval_action in enumerate(self.interval_actions):
            if interval_action.isChecked():
                interval, name = self.supported_intervals[i]
                return interval

    @property
    def repeat_animation(self) -> bool:
        return self.repeat_animation_action.isChecked()

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

    def show_menu(self, pos: QtCore.QPoint):
        self.edit_figure_action.setEnabled(self._animation is None and self._last_result is not None)
        self.save_figure_action.setEnabled(self._animation is None and self._last_result is not None)
        self.save_animation_action.setEnabled(self._animation is not None)
        self.menu.popup(QtGui.QCursor.pos())

    def show_chart(self, result: Union[ArtificialSample, SSUResult]):
        self._last_result = result
        self._axes.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        x = self.transfer(result.classes_phi)
        self._axes.set_title(result.name)
        self._axes.set_xlabel(self.xlabel)
        self._axes.set_ylabel(self.ylabel)
        self._axes.plot(x, result.sample.distribution, c="#ffffff00", marker=".", ms=3, mfc=normal_color(),
                        mec=normal_color(), label="Observation")
        self._axes.set_xlim(x[0], x[-1])
        self._axes.set_ylim(0.0, round(np.max(result.sample.distribution) * 1.2, 2))
        lmse_loss = loss_numpy("lmse")(result.distribution, result.sample.distribution, None)
        self._axes.plot(x, result.distribution, c=normal_color(), label=f"Prediction (LMSE={lmse_loss:.2f})")
        for i, component in enumerate(result):
            mode_micron = mode(result.classes, result.classes_phi, component.distribution, is_geometric=True)
            self._axes.plot(x, component.distribution * component.proportion, c=plt.get_cmap()(i),
                            label=f"C{i + 1} ({mode_micron:.2f} μm, {component.proportion:.2%}))")
        if self.xlog:
            self._axes.set_xscale("log")
        if self.show_mode:
            modes = [self.transfer(mode(result.classes, result.classes_phi, component.distribution,
                                        is_geometric=False)) for component in result]
            colors = [plt.get_cmap()(i) for i in range(len(result))]
            self._axes.vlines(modes, 0.0, 1.0, colors=colors)
        if self.show_legend:
            self._axes.legend(loc="upper left")
        self._figure.tight_layout()
        self._canvas.draw()

    def show_animation(self, result: SSUResult):
        assert isinstance(result, SSUResult)
        assert result.n_iterations > 1
        self._last_result = result
        self._axes.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        x = self.transfer(result.classes_phi)
        if self.xlog:
            self._axes.set_xscale("log")
        self._axes.set_title(result.name)
        self._axes.set_xlabel(self.xlabel)
        self._axes.set_ylabel(self.ylabel)
        observation_line = self._axes.plot(x, result.sample.distribution, c="#ffffff00", marker=".", ms=3,
                                           mfc=normal_color(), mec=normal_color(), label="Observation")[0]
        self._axes.set_xlim(x[0], x[-1])
        self._axes.set_ylim(0.0, round(np.max(result.sample.distribution) * 1.2, 2))

        prediction_line: Optional[plt.Line2D] = None
        component_lines: List[plt.Line2D] = []
        mode_lines: Optional[plt.Artist] = None
        legend: Optional[plt.Artist] = None

        def init():
            nonlocal prediction_line
            nonlocal component_lines
            nonlocal mode_lines
            nonlocal legend
            if prediction_line is None:
                lmse_loss = loss_numpy("lmse")(result.distribution, result.sample.distribution, None)
                prediction_line = self._axes.plot(x, result.distribution, c=normal_color(),
                                                  label=f"Prediction (LMSE={lmse_loss:.2f})")[0]
                for i, component in enumerate(result):
                    mode_micron = mode(result.classes, result.classes_phi, component.distribution, is_geometric=True)
                    line = self._axes.plot(x, component.distribution * component.proportion, c=plt.get_cmap()(i),
                                           label=f"C{i + 1} ({mode_micron:.2f} μm, {component.proportion:.2%}))")[0]
                    component_lines.append(line)
                if self.show_mode:
                    modes = [self.transfer(mode(result.classes, result.classes_phi, component.distribution,
                                                is_geometric=False)) for component in result]
                    colors = [plt.get_cmap()(i) for i in range(len(result))]
                    mode_lines = self._axes.vlines(modes, 0.0, 1.0, colors=colors)
                if self.show_legend:
                    legend = self._axes.legend(loc="upper left")
            artists = [prediction_line, *component_lines]
            if mode_lines is not None:
                artists.append(mode_lines)
            if legend is not None:
                artists.append(legend)
            return artists

        def animate(current: SSUResult):
            nonlocal prediction_line
            nonlocal component_lines
            nonlocal mode_lines
            nonlocal legend
            prediction_line.set_ydata(current.distribution)
            for i, (line, component) in enumerate(zip(component_lines, current)):
                mode_micron = mode(current.classes, current.classes_phi, component.distribution, is_geometric=True)
                line.set_ydata(component.distribution * component.proportion)
                line.set_label(f"C{i + 1} ({mode_micron:.2f} μm, {component.proportion:.2%}))")
            artists = [prediction_line, *component_lines]
            if self.show_mode:
                mode_lines.remove()
                modes = [self.transfer(mode(current.classes, current.classes_phi, component.distribution,
                                            is_geometric=False)) for component in current]
                colors = [plt.get_cmap()(i) for i in range(len(current))]
                mode_lines = self._axes.vlines(modes, 0.0, 1.0, colors=colors)
                artists.append(mode_lines)
            if self.show_legend:
                lmse_loss = loss_numpy("lmse")(current.distribution, current.sample.distribution, None)
                handles = [observation_line, prediction_line, *component_lines]
                labels = ["Observation", f"Prediction (LMSE={lmse_loss:.2f})"]
                for i, component in enumerate(current):
                    mode_micron = mode(current.classes, current.classes_phi, component.distribution, is_geometric=True)
                    label = f"C{i + 1} ({mode_micron:.2f} μm, {component.proportion:.2%})"
                    labels.append(label)
                legend = self._axes.legend(handles=handles, labels=labels, loc="upper left")
                artists.append(legend)
            return artists

        self._animation = FuncAnimation(self._figure, animate, frames=result.history, init_func=init,
                                        interval=self.animation_interval, blit=True, repeat=self.repeat_animation,
                                        repeat_delay=5.0, save_count=result.n_iterations)

    def show_result(self, result: Union[ArtificialSample, SSUResult]):
        if self.animated and isinstance(result, SSUResult) and result.n_iterations > 1:
            self.show_animation(result)
        else:
            self.show_chart(result)

    def update_chart(self):
        self._figure.clear()
        self._axes = self._figure.subplots()
        if self._last_result is not None:
            self.show_result(self._last_result)

    def retranslate(self):
        super().retranslate()
        self.scale_menu.setTitle(self.tr("Scale"))
        for action, (key, name) in zip(self.scale_actions, self.supported_scales):
            action.setText(name)
        self.show_mode_lines_action.setText(self.tr("Show Mode Lines"))
        self.show_legend_action.setText(self.tr("Show Legend"))
        self.animated_action.setText(self.tr("Animated"))
        self.interval_menu.setTitle(self.tr("Animation Interval"))
        for action, (interval, name) in zip(self.interval_actions, self.supported_intervals):
            action.setText(name)
        self.repeat_animation_action.setText(self.tr("Repeat Animation"))
        self.save_animation_action.setText(self.tr("Save Animation"))
