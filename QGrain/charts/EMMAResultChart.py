__all__ = ["EMMAResultChart"]

from typing import *

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.animation import FuncAnimation
from numpy import ndarray

from . import BaseChart
from . import normal_color
from ..models import EMMAResult
from ..statistics import to_microns
from ..utils import get_image_by_proportions


class EMMAResultChart(BaseChart):
    N_DISPLAY_SAMPLES = 200

    def __init__(self, parent=None, figsize=(4.4, 4.4)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("EMMA Chart"))
        # self.axes = self.figure.subplots()
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
        self.loss_menu = QtWidgets.QMenu(self.tr("Loss Function"))
        self.menu.insertMenu(self.edit_figure_action, self.loss_menu)
        self.loss_group = QtGui.QActionGroup(self.loss_menu)
        self.loss_group.setExclusive(True)
        self.loss_actions: List[QtGui.QAction] = []
        for key, name in self.supported_losses:
            loss_action = self.loss_group.addAction(name)
            loss_action.setCheckable(True)
            loss_action.triggered.connect(self.update_chart)
            self.loss_menu.addAction(loss_action)
            self.loss_actions.append(loss_action)
        self.loss_actions[8].setChecked(True)
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
        self.repeat_action = QtGui.QAction(self.tr("Repeat Animation"))
        self.repeat_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.repeat_action)
        self.repeat_action.setCheckable(True)
        self.repeat_action.setChecked(False)
        self._last_result: Optional[EMMAResult] = None

    @property
    def supported_scales(self) -> Sequence[Tuple[str, str]]:
        scales = (("log-linear", self.tr("Log-linear")),
                  ("log", self.tr("Log")),
                  ("phi", self.tr("Phi")),
                  ("linear", self.tr("Linear")))
        return scales

    @property
    def supported_losses(self) -> Sequence[Tuple[str, str]]:
        losses = (("1-norm", self.tr("1 Norm")),
                  ("2-norm", self.tr("2 Norm")),
                  ("3-norm", self.tr("3 Norm")),
                  ("4-norm", self.tr("4 Norm")),
                  ("mae", self.tr("MAE")),
                  ("mse", self.tr("MSE")),
                  ("rmse", self.tr("RMSE")),
                  ("rmlse", self.tr("RMLSE")),
                  ("lmse", self.tr("LMSE")),
                  ("cosine", self.tr("Cosine")),
                  ("angular", self.tr("Angular")))
        return losses

    @property
    def supported_intervals(self) -> Sequence[Tuple[int, str]]:
        intervals = ((5, self.tr("5 ms")),
                     (10, self.tr("10 ms")),
                     (20, self.tr("20 ms")),
                     (30, self.tr("30 ms")),
                     (60, self.tr("60 ms")))
        return intervals

    @property
    def scale(self) -> str:
        for i, scale_action in enumerate(self.scale_actions):
            if scale_action.isChecked():
                key, name = self.supported_scales[i]
                return key

    @property
    def loss(self) -> Tuple[str, str]:
        for i, distance_action in enumerate(self.loss_actions):
            if distance_action.isChecked():
                key, name = self.supported_losses[i]
                return (key, name)

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

    def show_menu(self, pos: QtCore.QPoint):
        self.edit_figure_action.setEnabled(self._last_result is not None and not self.animated)
        self.save_figure_action.setEnabled(self._last_result is not None and not self.animated)
        self.menu.popup(QtGui.QCursor.pos())

    def show_chart(self, result: EMMAResult):
        self._last_result = result
        self._figure.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        classes = self.transfer(result.dataset.classes_phi)
        loss_key, loss_name = self.loss
        loss_series = result.loss_series(loss_key)
        iteration_indexes = np.linspace(1, len(loss_series), len(loss_series))
        interval = max(1, result.n_samples // self.N_DISPLAY_SAMPLES)
        sample_axes = self._figure.add_subplot(2, 2, 1)
        for sample in result.dataset[::interval]:
            sample_axes.plot(classes, sample.distribution*100, c=normal_color(), alpha=0.2)
        if self.xlog:
            sample_axes.set_xscale("log")
        sample_axes.set_xlim(classes[0], classes[-1])
        sample_axes.set_ylim(0.0, round(np.max(result.dataset.distributions) * 1.2, 2)*100)
        sample_axes.set_xlabel(self.x_label)
        sample_axes.set_ylabel(self.y_label)
        sample_axes.set_title(self.tr("GSDs"))
        loss_axes = self._figure.add_subplot(2, 2, 2)
        loss_axes.plot(iteration_indexes, loss_series, c=normal_color())
        loss_axes.set_xlim(0, len(loss_series))
        loss_axes.set_xlabel(self.tr("Iteration"))
        loss_axes.set_ylabel(loss_name)
        loss_axes.set_title(self.tr("Loss variation"))
        end_member_axes = self._figure.add_subplot(2, 2, 3)
        if self.xlog:
            end_member_axes.set_xscale("log")
        for i in range(result.n_members):
            end_member_axes.plot(classes, result.end_members[i]*100, c=plt.get_cmap()(i),
                                 label=r"$\rm EM_{0}$".format(i+1), zorder=10 + i)
        end_member_axes.set_xlim(classes[0], classes[-1])
        end_member_axes.set_ylim(0.0, round(np.max(result.end_members) * 1.2, 2)*100)
        end_member_axes.set_xlabel(self.x_label)
        end_member_axes.set_ylabel(self.y_label)
        end_member_axes.set_title(self.tr("End members"))
        proportion_axes = self._figure.add_subplot(2, 2, 4)
        image = get_image_by_proportions(result.proportions, resolution=100)
        proportion_axes.imshow(image, plt.get_cmap(), aspect="auto", vmin=0, vmax=9,
                               extent=(0.0, result.n_samples, 100, 0.0), interpolation="none")
        proportion_axes.set_xlim(0, result.n_samples)
        proportion_axes.set_ylim(0, 100)
        proportion_axes.set_yticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100"])
        proportion_axes.set_xlabel(self.tr("Sample index"))
        proportion_axes.set_ylabel(self.tr("Proportion ({0})").format(r"$\%$"))
        proportion_axes.set_title(self.tr("Proportions"))
        self._figure.tight_layout()
        self._canvas.draw()

    def show_animation(self, result: EMMAResult):
        assert result.n_iterations > 1
        self._last_result = result
        self._figure.clear()
        if self._animation is not None:
            self._animation._stop()
            self._animation = None
        classes = self.transfer(result.dataset.classes_phi)
        loss_key, loss_name = self.loss
        loss_series = result.loss_series(loss_key)
        iteration_indexes = np.linspace(1, len(loss_series), len(loss_series))
        loss_series = result.loss_series(self.loss[0])
        min_distance, max_distance = np.min(loss_series), np.max(loss_series)
        interval = max(1, result.n_samples // self.N_DISPLAY_SAMPLES)
        sample_axes = self._figure.add_subplot(2, 2, 1)
        for sample in result.dataset[::interval]:
            sample_axes.plot(classes, sample.distribution*100, c=normal_color(), alpha=0.2)
        if self.xlog:
            sample_axes.set_xscale("log")
        sample_axes.set_xlim(classes[0], classes[-1])
        sample_axes.set_ylim(0.0, round(np.max(result.dataset.distributions) * 1.2, 2)*100)
        sample_axes.set_xlabel(self.x_label)
        sample_axes.set_ylabel(self.y_label)
        sample_axes.set_title(self.tr("GSDs"))
        loss_axes = self._figure.add_subplot(2, 2, 2)
        loss_axes.plot(iteration_indexes, loss_series, c=normal_color())
        loss_axes.set_xlim(0, len(loss_series))
        loss_axes.set_xlabel(self.tr("Iteration"))
        loss_axes.set_ylabel(loss_name)
        loss_axes.set_title(self.tr("Loss variation"))
        end_member_axes = self._figure.add_subplot(2, 2, 3)
        if self.xlog:
            end_member_axes.set_xscale("log")
        end_member_axes.set_xlim(classes[0], classes[-1])
        end_member_axes.set_ylim(0, round(np.max(result.end_members) * 1.2, 2)*100)
        end_member_axes.set_xlabel(self.x_label)
        end_member_axes.set_ylabel(self.y_label)
        end_member_axes.set_title(self.tr("End members"))
        proportion_axes = self._figure.add_subplot(2, 2, 4)
        proportion_axes.set_xlim(0, result.n_samples)
        proportion_axes.set_ylim(0, 100)
        proportion_axes.set_yticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100"])
        proportion_axes.set_xlabel(self.tr("Sample index"))
        proportion_axes.set_ylabel(self.tr("Proportion ({0})").format(r"$\%$"))
        proportion_axes.set_title(self.tr("Proportions"))

        iteration_line: Optional[plt.Line2D] = None
        end_member_curves: List[plt.Line2D] = []
        proportion_image: Optional[plt.Artist] = None

        def init():
            nonlocal iteration_line
            nonlocal end_member_curves
            nonlocal proportion_image
            if iteration_line is None:
                iteration_line = loss_axes.plot([1, 1], [min_distance, max_distance], c=normal_color())[0]
                for i in range(result.n_members):
                    curve = end_member_axes.plot(classes, result.end_members[i]*100,
                                                 c=plt.get_cmap()(i), label=r"$\rm EM_{0}$".format(i+1))[0]
                    end_member_curves.append(curve)
                image = get_image_by_proportions(result.proportions, resolution=100)
                proportion_image = proportion_axes.imshow(
                    image, plt.get_cmap(), aspect="auto", vmin=0, vmax=9,
                    extent=(0.0, result.n_samples, 100, 0.0), interpolation="none")
            return iteration_line, proportion_image, *end_member_curves

        def animate(args: Tuple[int, EMMAResult]):
            nonlocal iteration_line
            nonlocal end_member_curves
            nonlocal proportion_image
            iteration, current = args
            iteration_line.set_xdata((iteration, iteration))
            for i in range(current.n_members):
                end_member_curves[i].set_ydata(current.end_members[i]*100)
            image = get_image_by_proportions(current.proportions, resolution=100)
            proportion_image.set_data(image)
            return iteration_line, proportion_image, *end_member_curves

        self._animation = FuncAnimation(self._figure, animate, init_func=init, frames=enumerate(result.history),
                                        interval=self.animation_interval, blit=True, repeat=self.repeat_animation,
                                        repeat_delay=5.0, save_count=result.n_iterations)

    def show_result(self, result: EMMAResult):
        if self.animated and result.n_iterations > 1:
            self.show_animation(result)
        else:
            self.show_chart(result)

    def update_chart(self):
        if self._last_result is not None:
            self.show_result(self._last_result)

    def retranslate(self):
        self.setWindowTitle(self.tr("EMMA Chart"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.configure_subplots_action.setText(self.tr("Configure Subplots"))
        self.save_figure_action.setText(self.tr("Save Figure"))
        self.scale_menu.setTitle(self.tr("Scale"))
        for action, (key, name) in zip(self.scale_actions, self.supported_scales):
            action.setText(name)
        self.loss_menu.setTitle(self.tr("Loss Function"))
        for action, (key, name) in zip(self.loss_actions, self.supported_losses):
            action.setText(name)
        self.animated_action.setText(self.tr("Animated"))
        self.interval_menu.setTitle(self.tr("Animation Interval"))
        for action, (interval, name) in zip(self.interval_actions, self.supported_intervals):
            action.setText(name)
        self.repeat_action.setText(self.tr("Repeat Animation"))
