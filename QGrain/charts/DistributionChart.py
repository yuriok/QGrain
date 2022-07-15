__all__ = ["DistributionChart"]

from typing import *

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.animation import FuncAnimation
from numpy import ndarray
from scipy.stats import pearsonr

from . import BaseChart
from . import normal_color
from ..models import SSUResult, ArtificialSample
from ..statistics import to_microns, mode


class DistributionChart(BaseChart):
    def __init__(self, parent=None, size=(3, 2.5)):
        super().__init__(parent=parent, figsize=size)
        self._axes: plt.Axes = self._figure.subplots()
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
        self.scale_actions[0].setChecked(True)
        self.show_mode_lines_action: QtGui.QAction = QtGui.QAction(self.tr("Show Mode Lines"))
        self.show_mode_lines_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.show_mode_lines_action)
        self.show_mode_lines_action.setCheckable(True)
        self.show_mode_lines_action.setChecked(False)
        self.show_legend_action: QtGui.QAction = QtGui.QAction(self.tr("Show Legend"))
        self.show_legend_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.show_legend_action)
        self.show_legend_action.setCheckable(True)
        self.show_legend_action.setChecked(False)
        self.animated_action: QtGui.QAction = QtGui.QAction(self.tr("Animated"))
        self.animated_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.animated_action)
        self.animated_action.setCheckable(True)
        self.animated_action.setChecked(False)
        self.interval_menu: QtWidgets.QMenu = QtWidgets.QMenu(self.tr("Animation Interval"))
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
        self.repeat_animation_action: QtGui.QAction = QtGui.QAction(self.tr("Repeat Animation"))
        self.repeat_animation_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.repeat_animation_action)
        self.repeat_animation_action.setCheckable(True)
        self.repeat_animation_action.setChecked(False)
        self.save_animation_action: QtGui.QAction = QtGui.QAction(self.tr("Save Animation"))
        self.menu.addAction(self.save_animation_action)
        self.save_animation_action.triggered.connect(self.save_animation)
        self._last_result: Union[ArtificialSample, SSUResult, None] = None

    @property
    def supported_scales(self) -> Tuple[Tuple[str, str]]:
        scales = (("log-linear", self.tr("Log-linear")),
                  ("log", self.tr("Log")),
                  ("phi", self.tr("Phi")),
                  ("linear", self.tr("Linear")))
        return scales

    @property
    def supported_intervals(self) -> Tuple[Tuple[int, str]]:
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
    def show_mode_lines(self) -> bool:
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

    def show_chart(self, result: Union[ArtificialSample, SSUResult], quick=False):
        modes_phi = [mode(result.classes, result.classes_phi, component.distribution, is_geometric=False) for
                     component in result]
        if not quick:
            self._last_result = result
            self._figure.clear()
            if self._animation is not None:
                self._animation._stop()
                self._animation = None
            x = self.transfer(result.classes_phi)
            self._axes = self._figure.subplots()
            self._axes.set_title(result.name)
            self._axes.set_xlabel(self.xlabel)
            self._axes.set_ylabel(self.ylabel)
            self.observation_line = self._axes.plot(
                x, result.sample.distribution,
                c="#ffffff00",
                marker=".", ms=8,
                mfc=normal_color(), mec=normal_color(),
                label="Observation")[0]
            self._axes.set_xlim(x[0], x[-1])
            self._axes.set_ylim(0.0, round(np.max(result.sample.distribution) * 1.2, 2))
            self.prediction_line = self._axes.plot(x, result.distribution, c=normal_color(), label="Prediction")[0]
            self.component_lines = []
            for i, component in enumerate(result):
                component = self._axes.plot(x, component.distribution * component.proportion,
                                            c=plt.get_cmap()(i), label=f"C{i + 1}")[0]
                self.component_lines.append(component)
            self._figure.tight_layout()
        else:
            self.observation_line.set_ydata(result.sample.distribution)
            self.prediction_line.set_ydata(result.distribution)
            for component_line, component in zip(self.component_lines, result):
                component_line.set_ydata(component.distribution * component.proportion)
        if self.xlog:
            self._axes.set_xscale("log")
        if self.show_mode_lines:
            if hasattr(self, "vlines"):
                self.vlines.remove()
            modes = [self.transfer(mode_phi) for mode_phi in modes_phi]
            colors = [plt.get_cmap()(i) for i in range(len(result))]
            self.vlines = self._axes.vlines(modes, 0.0, 1.0, colors=colors)
        if self.show_legend:
            r, p = pearsonr(result.distribution, result.sample.distribution)
            r2 = r ** 2
            handles = [self.observation_line, self.prediction_line]
            handles.extend(self.component_lines)
            labels = ["Observation", f"Prediction ($R^2$={r2:.2f})"]
            for i, (mode_phi, component) in enumerate(zip(modes_phi, result)):
                mode_micron = to_microns(mode_phi)
                label = f"C{i + 1} ({mode_micron:.2f} μm, {component.proportion:.2%})"
                labels.append(label)
            self.legend = self._axes.legend(
                handles=handles, labels=labels,
                loc="upper left", prop={"size": 8})
        self._canvas.draw()

    def show_animation(self, result: SSUResult):
        self._last_result = result
        self._figure.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        self._axes = self._figure.subplots()
        results = iter(result.history)
        x = self.transfer(result.classes_phi)
        if self.xlog:
            self._axes.set_xscale("log")
        self._axes.set_title(result.name)
        self._axes.set_xlabel(self.xlabel)
        self._axes.set_ylabel(self.ylabel)
        self.observation_line = self._axes.plot(
            x, result.sample.distribution, c="#ffffff00", marker=".", ms=8,
            mfc=normal_color(), mec=normal_color(), label="Observation")[0]
        self._axes.set_xlim(x[0], x[-1])
        self._axes.set_ylim(0.0, round(np.max(result.distribution) * 1.2, 2))
        self._figure.tight_layout()

        def common(result: SSUResult):
            modes_phi = [mode(result.classes, result.classes_phi, component.distribution, is_geometric=False) for
                         component in result]
            if self.show_mode_lines:
                if hasattr(self, "vlines"):
                    self.vlines.remove()
                modes = [self.transfer(mode_phi) for mode_phi in modes_phi]
                colors = [plt.get_cmap()(i) for i in range(len(result))]
                self.vlines = self._axes.vlines(modes, 0.0, 1.0, colors=colors)
            if self.show_legend:
                r, p = pearsonr(result.distribution, result.sample.distribution)
                r2 = r ** 2
                handles = [self.observation_line, self.prediction_line]
                handles.extend(self.component_lines)
                labels = ["Target", f"Mixed ($R^2$={r2:.2f})"]
                for i, (mode_phi, component) in enumerate(zip(modes_phi, result)):
                    mode_micron = to_microns(mode_phi)
                    label = f"C{i + 1} ({mode_micron:.2f} μm, {component.proportion:.2%})"
                    labels.append(label)
                self.legend = self._axes.legend(
                    handles=handles, labels=labels,
                    loc="upper left", prop={"size": 8})

        def init():
            self.prediction_line = self._axes.plot(x, result.distribution, c=normal_color(), label="mixed")[0]
            self.component_lines = [
                self._axes.plot(x, component.distribution * component.proportion, c=plt.get_cmap()(i),
                                label=f"C{i + 1}")[0] for i, component in enumerate(result)]
            common(result)
            check_artists = [self.prediction_line]
            check_artists.extend(self.component_lines)
            if self.show_mode_lines:
                check_artists.append(self.vlines)
            if self.show_legend:
                check_artists.append(self.legend)
            return check_artists

        def animate(current: SSUResult):
            self.prediction_line.set_ydata(current.distribution)
            for line, component in zip(self.component_lines, current):
                line.set_ydata(component.distribution * component.proportion)
            common(current)
            check_artists = [self.prediction_line]
            check_artists.extend(self.component_lines)
            if self.show_mode_lines:
                check_artists.append(self.vlines)
            if self.show_legend:
                check_artists.append(self.legend)
            return check_artists

        self._animation = FuncAnimation(
            self._figure, animate, frames=results, init_func=init,
            interval=self.animation_interval, blit=True,
            repeat=self.repeat_animation, repeat_delay=3.0, save_count=result.n_iterations)

    def show_result(self, result: SSUResult):
        self._last_result = result
        self._figure.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        if self.animated:
            self.show_animation(result)
        else:
            self.show_chart(result)
        self._last_result = result

    def update_chart(self):
        self._figure.clear()
        self._axes = self._figure.subplots()
        if self._last_result is not None:
            self.show_result(self._last_result)
        elif self._last_result is not None:
            self.show_chart(self._last_result)

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
