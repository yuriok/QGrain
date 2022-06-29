__all__ = ["DistributionChart"]

import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.stats import pearsonr

from ..ssu import SSUResult, SSUViewModel
from ..statistics import convert_φ_to_μm, get_mode
from .BaseChart import BaseChart
from .config_matplotlib import normal_color

class DistributionChart(BaseChart):
    def __init__(self, parent=None, figsize=(4, 4)):
        super().__init__(parent=parent, figsize=figsize)
        self.axes = self.figure.subplots() # type: plt.Axes

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
        self.scale_actions[0].setChecked(True)
        self.show_mode_lines_action = QtGui.QAction(self.tr("Show Mode Lines")) # type: QtGui.QAction
        self.show_mode_lines_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.show_mode_lines_action)
        self.show_mode_lines_action.setCheckable(True)
        self.show_mode_lines_action.setChecked(False)
        self.show_legend_action = QtGui.QAction(self.tr("Show Legend")) # type: QtGui.QAction
        self.show_legend_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.show_legend_action)
        self.show_legend_action.setCheckable(True)
        self.show_legend_action.setChecked(False)
        self.animated_action = QtGui.QAction(self.tr("Animated")) # type: QtGui.QAction
        self.animated_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.animated_action)
        self.animated_action.setCheckable(True)
        self.animated_action.setChecked(False)
        self.interval_menu = QtWidgets.QMenu(self.tr("Animation Interval")) # type: QtWidgets.QMenu
        self.menu.insertMenu(self.edit_figure_action, self.interval_menu)
        self.interval_group = QtGui.QActionGroup(self.interval_menu)
        self.interval_group.setExclusive(True)
        self.interval_actions = [] # type: list[QtGui.QAction]
        for interval, name in self.supported_intervals:
            interval_action = self.interval_group.addAction(name)
            interval_action.setCheckable(True)
            interval_action.triggered.connect(self.update_chart)
            self.interval_menu.addAction(interval_action)
            self.interval_actions.append(interval_action)
        self.interval_actions[3].setChecked(True)
        self.repeat_animation_action = QtGui.QAction(self.tr("Repeat Animation")) # type: QtGui.QAction
        self.repeat_animation_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.repeat_animation_action)
        self.repeat_animation_action.setCheckable(True)
        self.repeat_animation_action.setChecked(False)
        self.save_animation_action = QtGui.QAction(self.tr("Save Animation")) # type: QtGui.QAction
        self.menu.addAction(self.save_animation_action)
        self.save_animation_action.triggered.connect(self.save_animation)

        self.animation = None
        self.last_model = None # type: SSUViewModel
        self.last_result = None # type: SSUResult
        self.file_dialog = QtWidgets.QFileDialog(parent=self)

    @property
    def supported_scales(self) -> typing.List[typing.Tuple[str, str]]:
        scales = [("log-linear", self.tr("Log-linear")),
                  ("log", self.tr("Log")),
                  ("phi", self.tr("Phi")),
                  ("linear", self.tr("Linear"))]
        return scales

    @property
    def supported_intervals(self) -> typing.List[typing.Tuple[int, str]]:
        intervals = [(5, self.tr("5 Milliseconds")),
                     (10, self.tr("10 Milliseconds")),
                     (20, self.tr("20 Milliseconds")),
                     (30, self.tr("30 Milliseconds")),
                     (60, self.tr("60 Milliseconds"))]
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

    @property
    def xlog(self) -> bool:
        if self.scale == "log-linear":
            return True
        else:
            return False

    def show_menu(self, pos: QtCore.QPoint):
        self.save_figure_action.setEnabled(self.last_model is not None or (self.last_result is not None and not self.animated))
        self.save_animation_action.setEnabled(self.last_result is not None and self.animated)
        self.menu.popup(QtGui.QCursor.pos())

    def show_model(self, model: SSUViewModel, quick=False):
        modes_φ = [model.classes_φ[np.unravel_index(np.argmax(distribution), distribution.shape)] for distribution in model.distributions]
        if self.animation is not None:
            self.animation._stop()
            self.animation = None
        if not quick:
            self.last_result = None
            self.last_model = model
            self.axes.clear()
            x = self.transfer(model.classes_φ)
            self.axes.set_title(model.title)
            self.axes.set_xlabel(self.xlabel)
            self.axes.set_ylabel(self.ylabel)
            self.target_line = self.axes.plot(x, model.target, c="#ffffff00", marker=".", ms=8, mfc=normal_color(), mew=0.0)[0]
            self.axes.set_xlim(x[0], x[-1])
            self.axes.set_ylim(0.0, round(np.max(model.target)*1.2, 2))
            self.mixed_line = self.axes.plot(x, model.mixed, c=normal_color())[0]
            self.component_lines = []
            for i, (distribution, proportion) in enumerate(zip(model.distributions, model.proportions)):
                component = self.axes.plot(x, distribution*proportion, c=plt.get_cmap()(i))[0]
                self.component_lines.append(component)
            self.figure.tight_layout()
        else:
            self.target_line.set_ydata(model.target)
            self.mixed_line.set_ydata(model.mixed)
            for component_line, distribution, proportion in zip(self.component_lines, model.distributions, model.proportions):
                component_line.set_ydata(distribution*proportion)
        if self.xlog:
            self.axes.set_xscale("log")
        if self.show_mode_lines:
            if hasattr(self, "vlines"):
                self.vlines.remove()
            modes = [self.transfer(mode_φ) for mode_φ in modes_φ]
            colors = [plt.get_cmap()(i) for i in range(model.n_components)]
            self.vlines = self.axes.vlines(modes, 0.0, 1.0, colors=colors)
        if self.show_legend:
            r, p = pearsonr(model.target, model.mixed)
            r2 = r**2
            handles = [self.target_line, self.mixed_line]
            handles.extend(self.component_lines)
            labels = ["Target", f"Mixed ($R^2$={r2:.2f})"]
            for i, (mode_φ, proportion) in enumerate(zip(modes_φ, model.proportions)):
                mode_μm = convert_φ_to_μm(mode_φ)
                label = f"{model.component_prefix}{i+1} ({mode_μm:.2f} μm, {proportion:.2%})"
                labels.append(label)
            self.legend = self.axes.legend(
                handles=handles, labels=labels,
                loc="upper left", prop={"size": 8})
        self.canvas.draw()

    def show_animation(self, result: SSUResult):
        if self.animation is not None:
            self.animation._stop()
            self.animation = None
        self.last_model = None
        self.last_result = result

        models = iter(result.view_models)
        first = next(models)
        x = self.transfer(first.classes_φ)
        self.axes.cla()
        if self.xlog:
            self.axes.set_xscale("log")
        self.axes.set_title(first.title)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.target_line = self.axes.plot(x, first.target, c="#ffffff00", marker=".", ms=8, mfc=normal_color(), mew=0.0)[0]
        self.axes.set_xlim(x[0], x[-1])
        self.axes.set_ylim(0.0, round(np.max(first.target)*1.2, 2))
        self.figure.tight_layout()
        # self.canvas.draw()
        def common(model: SSUViewModel):
            modes_φ = [model.classes_φ[np.unravel_index(np.argmax(distribution), distribution.shape)] for distribution in model.distributions]
            if self.show_mode_lines:
                if hasattr(self, "vlines"):
                    self.vlines.remove()
                modes = [self.transfer(mode_φ) for mode_φ in modes_φ]
                colors = [plt.get_cmap()(i) for i in range(model.n_components)]
                self.vlines = self.axes.vlines(modes, 0.0, 1.0, colors=colors)
            if self.show_legend:
                r, p = pearsonr(model.target, model.mixed)
                r2 = r**2
                handles = [self.target_line, self.mixed_line]
                handles.extend(self.component_lines)
                labels = ["Target", f"Mixed ($R^2$={r2:.2f})"]
                for i, (mode_φ, proportion) in enumerate(zip(modes_φ, model.proportions)):
                    mode_μm = convert_φ_to_μm(mode_φ)
                    label = f"{model.component_prefix}{i+1} ({mode_μm:.2f} μm, {proportion:.2%})"
                    labels.append(label)
                self.legend = self.axes.legend(
                    handles=handles, labels=labels,
                    loc="upper left", prop={"size": 8})
        def init():
            model = first
            self.mixed_line = self.axes.plot(x, model.mixed, c=normal_color())[0]
            self.component_lines = [self.axes.plot(x, distribution*proportion, c=plt.get_cmap()(i))[0] for i, (distribution, proportion) in enumerate(zip(model.distributions, model.proportions))]
            common(model)
            check_artists = [self.mixed_line]
            check_artists.extend(self.component_lines)
            if self.show_mode_lines:
                check_artists.append(self.vlines)
            if self.show_legend:
                check_artists.append(self.legend)
            return check_artists
        def animate(current: SSUViewModel):
            model = current
            self.mixed_line.set_ydata(current.mixed)
            for line, distribution, proportion in zip(self.component_lines, model.distributions, model.proportions):
                line.set_ydata(distribution*proportion)
            common(model)
            check_artists = [self.mixed_line]
            check_artists.extend(self.component_lines)
            if self.show_mode_lines:
                check_artists.append(self.vlines)
            if self.show_legend:
                check_artists.append(self.legend)
            return check_artists
        self.animation = FuncAnimation(
            self.figure, animate, frames=models, init_func=init,
            interval=self.animation_interval, blit=True,
            repeat=self.repeat_animation, repeat_delay=3.0, save_count=result.n_iterations)

    def show_result(self, result: SSUResult):
        if self.animated:
            self.show_animation(result)
        else:
            self.show_model(result.view_model)
        self.last_model = None
        self.last_result = result

    def save_animation(self):
        if self.last_result is not None:
            filename, format_str = self.file_dialog.getSaveFileName(
                self, self.tr("Choose a filename to save the animation of this SSU result"),
                None, "MPEG-4 Video File (*.mp4);;Graphics Interchange Format (*.gif)")
            if filename is None or filename == "":
                return
            progress = QtWidgets.QProgressDialog(self)
            progress.setWindowTitle("QGrain")
            progress.setRange(0, 100)
            progress.setLabelText(self.tr("Saving Animation [{0} Frames]").format(self.last_result.n_iterations))
            canceled = False
            def save_callback(i, n):
                if progress.wasCanceled():
                    nonlocal canceled
                    canceled = True
                    raise StopIteration()
                progress.setValue((i+1)/n*100)
                QtCore.QCoreApplication.processEvents()
            self.show_animation(self.last_result)
            # plt.rcParams["savefig.dpi"] = 120.0
            if "*.gif" in format_str:
                if not ImageMagickWriter.isAvailable():
                    self.normal_msg.setWindowTitle(self.tr("Error"))
                    self.normal_msg.setText(self.tr("ImageMagick is not installed, please download and install it from its offical website (https://imagemagick.org/index.php)."))
                    self.normal_msg.exec_()
                else:
                    self.animation.save(filename, writer="imagemagick", fps=30, progress_callback=save_callback)
            elif "*.mp4" in format_str:
                if not FFMpegWriter.isAvailable():
                    self.normal_msg.setWindowTitle(self.tr("Error"))
                    self.normal_msg.setText(self.tr("FFMpeg is not installed, please download and install it from its offical website (https://ffmpeg.org/)."))
                    self.normal_msg.exec_()
                else:
                    self.animation.save(filename, writer="ffmpeg", fps=30, progress_callback=save_callback)
            # plt.rcParams["savefig.dpi"] = 300.0
            if not canceled:
                progress.setValue(100)

    def update_chart(self):
        self.figure.clear()
        self.axes = self.figure.subplots()
        if self.last_model is not None:
            self.show_model(self.last_model)
        elif self.last_result is not None:
            self.show_result(self.last_result)

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
