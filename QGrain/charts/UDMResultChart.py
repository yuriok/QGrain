from typing import *

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.animation import FuncAnimation
from numpy import ndarray

from . import BaseChart
from . import normal_color
from ..models import UDMResult
from ..statistics import to_microns
from ..utils import get_image_by_proportions


def summarize(components: ndarray, q=0.01):
    mean = np.mean(components, axis=0)
    upper = np.quantile(components, q=1-q, axis=0)
    lower = np.quantile(components, q=q, axis=0)
    return mean, lower, upper


class UDMResultChart(BaseChart):
    N_DISPLAY_SAMPLES = 200

    def __init__(self, parent=None, figsize=(4.4, 4.4)):
        super().__init__(parent=parent, figsize=figsize)
        # self.axes = self.figure.subplots()
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
        self.repeat_action: QtGui.QAction = QtGui.QAction(self.tr("Repeat Animation"))
        self.repeat_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.repeat_action)
        self.repeat_action.setCheckable(True)
        self.repeat_action.setChecked(False)
        self.save_animation_action: QtGui.QAction = QtGui.QAction(self.tr("Save Animation"))
        self.menu.addAction(self.save_animation_action)
        self.save_animation_action.triggered.connect(self.save_animation)
        self._last_result = None

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
        return self.repeat_action.isChecked()

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
        self.edit_figure_action.setEnabled(self._last_result is not None and not self.animated)
        self.save_figure_action.setEnabled(self._last_result is not None and not self.animated)
        self.save_animation_action.setEnabled(self._last_result is not None and self.animated)
        self.menu.popup(QtGui.QCursor.pos())

    def show_chart(self, result: UDMResult):
        self._last_result = result
        self._figure.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        interval = max(1, result.n_samples//self.N_DISPLAY_SAMPLES)
        iteration_indexes = np.linspace(1, len(result.loss_series("total")), len(result.loss_series("total")))
        classes = self.transfer(result.dataset.classes_phi)
        sample_axes = self._figure.add_subplot(2, 2, 1)
        for sample in result.dataset[::interval]:
            sample_axes.plot(classes, sample.distribution, c=normal_color(), alpha=0.2)
        if self.xlog:
            sample_axes.set_xscale("log")
        sample_axes.set_xlim(classes[0], classes[-1])
        sample_axes.set_ylim(0.0, round(np.max(result.dataset.distributions)*1.2, 2))
        sample_axes.set_xlabel(self.xlabel)
        sample_axes.set_ylabel(self.ylabel)
        sample_axes.set_title("GSDs")
        loss_axes = self._figure.add_subplot(2, 2, 2)
        loss_axes.plot(result.loss_series("total"), color=plt.get_cmap()(0), label="Sum")
        loss_axes.plot(result.loss_series("distribution"), color=plt.get_cmap()(1), label="GSDs")
        loss_axes.plot(result.loss_series("component"), color=plt.get_cmap()(2), label="Components")
        loss_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        loss_axes.set_xlabel("Iteration")
        loss_axes.set_ylabel("Loss")
        loss_axes.set_title("Loss variation")
        loss_axes.legend(loc="upper right")
        component_axes = self._figure.add_subplot(2, 2, 3)
        mean, lower, upper = summarize(result.components, q=0.01)
        for i in range(result.n_components):
            component_axes.plot(classes, mean[i], c=plt.get_cmap()(i), zorder=20+i)
            component_axes.fill_between(
                classes, lower[i], upper[i],
                color=plt.get_cmap()(i),
                alpha=0.2, lw=0.02, zorder=10+i)
        if self.xlog:
            component_axes.set_xscale("log")
        component_axes.set_xlim(classes[0], classes[-1])
        component_axes.set_ylim(0.0, round(np.max(mean)*1.2, 2))
        component_axes.set_xlabel(self.xlabel)
        component_axes.set_ylabel(self.ylabel)
        component_axes.set_title("Components")
        proportion_axes = self._figure.add_subplot(2, 2, 4)
        image = get_image_by_proportions(result.proportions[:, 0, :], resolution=100)
        proportion_axes.imshow(image, plt.get_cmap(), aspect="auto", vmin=0, vmax=9)
        proportion_axes.set_xlim(-0.5, result.n_samples - 0.5)
        proportion_axes.set_ylim(-0.5, 99.5)
        proportion_axes.set_yticks([0, 20, 40, 60, 80, 100], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        proportion_axes.set_xlabel("Sample index")
        proportion_axes.set_ylabel("Proportion")
        proportion_axes.set_title("Proportions")
        self._figure.tight_layout()
        self._canvas.draw()

    def show_animation(self, result: UDMResult):
        self._last_result = result
        self._figure.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        interval = max(1, result.n_samples // self.N_DISPLAY_SAMPLES)
        iteration_indexes = np.linspace(1, len(result.loss_series("total")), len(result.loss_series("total")))
        classes = self.transfer(result.dataset.classes_phi)
        losses = np.array([series for key, series in result._loss_series.items()])
        min_distance, max_distance = np.min(losses), np.max(losses)
        sample_axes = self._figure.add_subplot(2, 2, 1)
        for sample in result.dataset[::interval]:
            sample_axes.plot(classes, sample.distribution, c=normal_color(), alpha=0.2)
        if self.xlog:
            sample_axes.set_xscale("log")
        sample_axes.set_xlim(classes[0], classes[-1])
        sample_axes.set_ylim(0.0, round(np.max(result.dataset.distributions)*1.2, 2))
        sample_axes.set_xlabel(self.xlabel)
        sample_axes.set_ylabel(self.ylabel)
        sample_axes.set_title("GSDs")

        loss_axes = self._figure.add_subplot(2, 2, 2)
        loss_axes.plot(result.loss_series("total"), color=plt.get_cmap()(0), label="Sum")
        loss_axes.plot(result.loss_series("distribution"), color=plt.get_cmap()(1), label="GSDs")
        loss_axes.plot(result.loss_series("component"), color=plt.get_cmap()(2), label="Components")
        loss_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        loss_axes.set_xlabel("Iteration")
        loss_axes.set_ylabel("Loss")
        loss_axes.set_title("Loss variation")
        loss_axes.legend(loc="upper right")
        component_axes = self._figure.add_subplot(2, 2, 3)
        mean, lower, upper = summarize(result.components, q=0.01)
        if self.xlog:
            component_axes.set_xscale("log")
        component_axes.set_xlim(classes[0], classes[-1])
        component_axes.set_ylim(0.0, round(np.max(mean)*1.2, 2))
        component_axes.set_xlabel(self.xlabel)
        component_axes.set_ylabel(self.ylabel)
        component_axes.set_title("Components")
        proportion_axes = self._figure.add_subplot(2, 2, 4)
        proportion_axes.set_xlim(-0.5, result.n_samples - 0.5)
        proportion_axes.set_ylim(-0.5, 99.5)
        proportion_axes.set_yticks([0, 20, 40, 60, 80, 100], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        proportion_axes.set_xlabel("Sample index")
        proportion_axes.set_ylabel("Proportion [%]")
        proportion_axes.set_title("Proportions")
        # self.figure.tight_layout()
        # self.canvas.draw()

        def init():
            self.iteration_line = loss_axes.plot([1, 1], [min_distance, max_distance], c=normal_color())[0]
            self.component_curves = []
            self.component_shadows = []
            mean, lower, upper = summarize(result.components, q=0.01)
            for i in range(result.n_components):
                component_curve = component_axes.plot(classes, mean[i], c=plt.get_cmap()(i), zorder=20+i)[0]
                component_shadow = component_axes.fill_between(
                    classes, lower[i], upper[i],
                    color=plt.get_cmap()(i),
                    alpha=0.2, lw=0.02, zorder=10+i)
                self.component_curves.append(component_curve)
                self.component_shadows.append(component_shadow)
            image = get_image_by_proportions(result.proportions[:, 0, :], resolution=100)
            self.proportion_image = proportion_axes.imshow(image, plt.get_cmap(), aspect="auto", vmin=0, vmax=9)
            return self.iteration_line, self.proportion_image, *(self.component_curves + self.component_shadows)

        def animate(args: Tuple[int, UDMResult]):
            iteration, current = args
            mean, lower, upper = summarize(current.components, q=0.01)
            self.iteration_line.set_xdata([iteration, iteration])
            for i in range(current.n_components):
                self.component_curves[i].set_ydata(mean[i])
                verts_lower = np.concatenate([np.expand_dims(classes, axis=1), np.expand_dims(lower[i], axis=1)], axis=1)
                verts_upper = np.concatenate([np.expand_dims(classes[::-1], axis=1), np.expand_dims(upper[i][::-1], axis=1)], axis=1)
                verts = np.concatenate([verts_lower, verts_upper], axis=0)
                self.component_shadows[i].set_verts([verts])
            image = get_image_by_proportions(current.proportions[:, 0, :], resolution=100)
            self.proportion_image.set_data(image)
            return self.iteration_line, self.proportion_image, *(self.component_curves + self.component_shadows)

        self._animation = FuncAnimation(
            self._figure, animate, init_func=init,
            frames=enumerate(result.history), interval=self.animation_interval,
            blit=True, repeat=self.repeat_animation,
            repeat_delay=3.0, save_count=result.n_iterations)

    def show_result(self, result: UDMResult):
        self._last_result = result
        self._figure.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        if self.animated:
            self.show_animation(result)
        else:
            self.show_chart(result)

    def update_chart(self):
        if self._last_result is not None:
            self.show_result(self._last_result)

    def retranslate(self):
        super().retranslate()
        self.scale_menu.setTitle(self.tr("Scale"))
        for action, (key, name) in zip(self.scale_actions, self.supported_scales):
            action.setText(name)
        self.animated_action.setText(self.tr("Animated"))
        self.interval_menu.setTitle(self.tr("Animation Interval"))
        for action, (interval, name) in zip(self.interval_actions, self.supported_intervals):
            action.setText(name)
        self.repeat_action.setText(self.tr("Repeat Animation"))
        self.save_animation_action.setText(self.tr("Save Animation"))
