import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter
from PySide6 import QtCore, QtGui, QtWidgets

from ..statistic import convert_φ_to_μm
from ..udm import UDMResult
from .BaseChart import BaseChart
from .config_matplotlib import normal_color


def summarize(proportions: np.ndarray, components: np.ndarray, q=0.01):
    n_samples, n_components, n_classes = components.shape
    EM_mean = np.mean(np.expand_dims(proportions.squeeze(1), 2).repeat(n_classes, axis=2) * components, axis=0)
    EM_mean /= np.expand_dims(np.sum(EM_mean, axis=1), axis=1).repeat(n_classes, axis=1)
    EM_upper = np.zeros((n_components, n_classes))
    EM_lower = np.zeros((n_components, n_classes))
    for i in range(n_components):
        key = np.greater(proportions[:, 0, i], 1e-3)
        EM_upper[i] = np.quantile(components[:, i, :][key], q=1-q, axis=0)
        EM_lower[i] = np.quantile(components[:, i, :][key], q=q, axis=0)
    return EM_mean, EM_lower, EM_upper


class UDMResultChart(BaseChart):
    N_DISPLAY_SAMPLES = 200
    def __init__(self, parent=None, figsize=(4, 6)):
        super().__init__(parent=parent, figsize=figsize)
        # self.axes = self.figure.subplots()
        self.scale_menu = QtWidgets.QMenu(self.tr("Scale")) # type: QtWidgets.QMenu
        self.menu.insertMenu(self.save_figure_action, self.scale_menu)
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

        self.animated_action = QtGui.QAction(self.tr("Animated")) # type: QtGui.QAction
        self.animated_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.save_figure_action, self.animated_action)
        self.animated_action.setCheckable(True)
        self.animated_action.setChecked(False)

        self.interval_menu = QtWidgets.QMenu(self.tr("Animation Interval")) # type: QtWidgets.QMenu
        self.menu.insertMenu(self.save_figure_action, self.interval_menu)
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

        self.repeat_action = QtGui.QAction(self.tr("Repeat Animation")) # type: QtGui.QAction
        self.repeat_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.save_figure_action, self.repeat_action)
        self.repeat_action.setCheckable(True)
        self.repeat_action.setChecked(False)

        self.save_animation_action = QtGui.QAction(self.tr("Save Animation")) # type: QtGui.QAction
        self.menu.addAction(self.save_animation_action)
        self.save_animation_action.triggered.connect(self.save_animation)

        self.__animation = None
        self.__last_result = None

        self.normal_msg = QtWidgets.QMessageBox(parent=self)
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

    def show_chart(self, result: UDMResult):
        interval = max(1, result.n_samples//self.N_DISPLAY_SAMPLES)
        sample_indexes = np.linspace(1, result.n_samples, result.n_samples)
        iteration_indexes = np.linspace(1, result.n_iterations, result.n_iterations)
        classes = self.transfer(result.dataset.classes_φ)
        GSDs = result.dataset.distribution_matrix
        sum_loss_series = result.distribution_loss_series + result.component_loss_series

        GSDs_axes = self.figure.add_subplot(2, 2, 1)
        for sample in result.dataset.samples[::interval]:
            GSDs_axes.plot(classes, sample.distribution, c=normal_color(), alpha=0.2)
        if self.xlog:
            GSDs_axes.set_xscale("log")
        GSDs_axes.set_xlim(classes[0], classes[-1])
        GSDs_axes.set_ylim(0.0, round(np.max(GSDs)*1.2, 2))
        GSDs_axes.set_xlabel(self.xlabel)
        GSDs_axes.set_ylabel(self.ylabel)
        GSDs_axes.set_title("GSDs")

        loss_axes = plt.subplot(2, 2, 2)
        loss_axes.plot(sum_loss_series, color=plt.get_cmap()(0), label="Sum")
        loss_axes.plot(result.distribution_loss_series, color=plt.get_cmap()(1), label="GSDs")
        loss_axes.plot(result.component_loss_series, color=plt.get_cmap()(2), label="Components")
        loss_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        loss_axes.set_xlabel("Iteration")
        loss_axes.set_ylabel("Loss")
        loss_axes.set_title("Loss variation")
        loss_axes.legend(loc="upper right", prop={"size": 8})

        component_axes = plt.subplot(2, 2, 3)
        mean, lower, upper = summarize(result.proportions, result.components, q=0.01)
        modes = [(i, result.dataset.classes_μm[np.unravel_index(np.argmax(mean[i]), mean[i].shape)]) for i in range(result.n_components)]
        # sort them by mode size
        modes.sort(key=lambda x: x[1])
        for i_c, (i, _) in enumerate(modes):
            component_axes.plot(classes, mean[i], c=plt.get_cmap()(i_c), zorder=20+i_c)
            component_axes.fill_between(
                classes, lower[i], upper[i],
                color=plt.get_cmap()(i_c),
                alpha=0.2, lw=0.02, zorder=10+i_c)
        if self.xlog:
            component_axes.set_xscale("log")
        component_axes.set_xlim(classes[0], classes[-1])
        component_axes.set_ylim(0.0, round(np.max(mean)*1.2, 2))
        component_axes.set_xlabel(self.xlabel)
        component_axes.set_ylabel(self.ylabel)
        component_axes.set_title("Components")

        proportion_axes = plt.subplot(2, 2, 4)
        bottom = np.zeros(result.n_samples)
        for i_em, (i, _) in enumerate(modes):
            proportion_axes.bar(
                sample_indexes[::interval],
                result.proportions[:, 0, i][::interval],
                bottom=bottom[::interval],
                width=interval, color=plt.get_cmap()(i_em))
            bottom += result.proportions[:, 0, i]
        proportion_axes.set_xlim(sample_indexes[0], sample_indexes[-1])
        proportion_axes.set_ylim(0.0, 1.0)
        proportion_axes.set_xlabel("Sample index")
        proportion_axes.set_ylabel("Proportion")
        proportion_axes.set_title("Proportions")
        self.figure.tight_layout()
        self.canvas.draw()

    def show_animation(self, result: UDMResult):
        interval = max(1, result.n_samples // self.N_DISPLAY_SAMPLES)
        sample_indexes = np.linspace(1, result.n_samples, result.n_samples)
        iteration_indexes = np.linspace(1, result.n_iterations, result.n_iterations)
        classes = self.transfer(result.dataset.classes_φ)
        GSDs = result.dataset.distribution_matrix
        sum_loss_series = result.distribution_loss_series + result.component_loss_series
        distances = np.array([result.distribution_loss_series, result.component_loss_series, sum_loss_series])
        min_distance, max_distance = np.min(distances), np.max(distances)

        GSDs_axes = self.figure.add_subplot(2, 2, 1)
        for sample in result.dataset.samples[::interval]:
            GSDs_axes.plot(classes, sample.distribution, c=normal_color(), alpha=0.2)
        if self.xlog:
            GSDs_axes.set_xscale("log")
        GSDs_axes.set_xlim(classes[0], classes[-1])
        GSDs_axes.set_ylim(0.0, round(np.max(GSDs)*1.2, 2))
        GSDs_axes.set_xlabel(self.xlabel)
        GSDs_axes.set_ylabel(self.ylabel)
        GSDs_axes.set_title("GSDs")

        loss_axes = plt.subplot(2, 2, 2)
        sum_loss_line = loss_axes.plot(sum_loss_series, color=plt.get_cmap()(0), label="Sum")[0]
        distribution_loss_line = loss_axes.plot(result.distribution_loss_series, color=plt.get_cmap()(1), label="GSDs")[0]
        component_loss_line = loss_axes.plot(result.component_loss_series, color=plt.get_cmap()(2), label="Components")[0]
        loss_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        loss_axes.set_xlabel("Iteration")
        loss_axes.set_ylabel("Loss")
        loss_axes.set_title("Loss variation")
        loss_axes.legend(
            handles=[sum_loss_line, distribution_loss_line, component_loss_line],
            labels=["Sum", "GSDs", "Components"],
            loc="upper right", prop={"size": 8})
        component_axes = plt.subplot(2, 2, 3)
        mean, lower, upper = summarize(result.proportions, result.components, q=0.01)
        modes = [(i, result.dataset.classes_μm[np.unravel_index(np.argmax(mean[i]), mean[i].shape)]) for i in range(result.n_components)]
        # sort them by mode size
        modes.sort(key=lambda x: x[1])
        if self.xlog:
            component_axes.set_xscale("log")
        component_axes.set_xlim(classes[0], classes[-1])
        component_axes.set_ylim(0.0, round(np.max(mean)*1.2, 2))
        component_axes.set_xlabel(self.xlabel)
        component_axes.set_ylabel(self.ylabel)
        component_axes.set_title("Components")

        proportion_axes = plt.subplot(2, 2, 4)
        proportion_axes.set_xlim(sample_indexes[0], sample_indexes[-1])
        proportion_axes.set_ylim(0.0, 1.0)
        proportion_axes.set_xlabel("Sample index")
        proportion_axes.set_ylabel("Proportion")
        proportion_axes.set_title("Proportions")
        # self.figure.tight_layout()
        # self.canvas.draw()

        def init():
            self.iteration_line = loss_axes.plot([1, 1], [min_distance, max_distance], c=normal_color())[0]
            self.component_curves = []
            self.component_shadows = []
            mean, lower, upper = summarize(result.proportions, result.components, q=0.01)
            for i_c, (i, _) in enumerate(modes):
                component_curve = component_axes.plot(classes, mean[i], c=plt.get_cmap()(i_c), zorder=20+i_c)[0]
                component_shadow = component_axes.fill_between(
                    classes, lower[i], upper[i],
                    color=plt.get_cmap()(i_c),
                    alpha=0.2, lw=0.02, zorder=10+i_c)
                self.component_curves.append(component_curve)
                self.component_shadows.append(component_shadow)

            bottom = np.zeros(result.n_samples)
            self.proportion_bars = []
            self.patches = []
            for i_em, (index, _) in enumerate(modes):
                bar = proportion_axes.bar(
                    sample_indexes[::interval],
                    result.proportions[:, 0, index][::interval],
                    bottom=bottom[::interval],
                    width=interval, color=plt.get_cmap()(i_em))
                self.proportion_bars.append(bar)
                self.patches.extend(bar.patches)
                bottom += result.proportions[:, 0, index]
            return self.iteration_line, *(self.component_curves + self.component_shadows + self.patches)

        def animate(args: typing.Tuple[int, UDMResult]):
            iteration, current = args
            mean, lower, upper = summarize(current.proportions, current.components, q=0.01)
            self.iteration_line.set_xdata([iteration, iteration])
            for i_em, (index, _) in enumerate(modes):
                self.component_curves[i_em].set_ydata(mean[index])
                verts_lower = np.concatenate([np.expand_dims(classes, axis=1), np.expand_dims(lower[i_em], axis=1)], axis=1)
                verts_upper = np.concatenate([np.expand_dims(classes[::-1], axis=1), np.expand_dims(upper[i_em][::-1], axis=1)], axis=1)
                verts = np.concatenate([verts_lower, verts_upper], axis=0)
                self.component_shadows[i_em].set_verts([verts])
            bottom = np.zeros(current.n_samples)
            for i_em, (index, _) in enumerate(modes):
                for rect, height, y in zip(self.proportion_bars[i_em].patches, current.proportions[:, 0, index][::interval], bottom[::interval]):
                    rect.set_height(height)
                    rect.set_y(y)
                bottom += current.proportions[:, 0, index]
            return self.iteration_line, *(self.component_curves + self.component_shadows + self.patches)

        self.__animation = FuncAnimation(
            self.figure, animate, init_func=init,
            frames=enumerate(result.history), interval=self.animation_interval,
            blit=True, repeat=self.repeat_animation,
            repeat_delay=3.0, save_count=result.n_iterations)

    def show_result(self, result: UDMResult):
        self.__last_result = result
        self.figure.clear()
        if self.__animation is not None:
            self.__animation._stop()
            self.__animation = None
        if self.animated:
            self.show_animation(result)
        else:
            self.show_chart(result)

    def update_chart(self):
        if self.__last_result is not None:
            self.show_result(self.__last_result)

    def save_animation(self):
        if self.__last_result is not None:
            filename, format_str  = self.file_dialog.getSaveFileName(
                self, self.tr("Choose a filename to save the animation of this UDM result"),
                None, "MPEG-4 Video File (*.mp4);;Graphics Interchange Format (*.gif)")
            if filename is None or filename == "":
                return
            progress = QtWidgets.QProgressDialog(self)
            progress.setWindowTitle("QGrain")
            progress.setRange(0, 100)
            progress.setLabelText(self.tr("Saving Animation [{0} Frames]").format(self.__last_result.n_iterations))
            canceled = False
            def save_callback(i, n):
                if progress.wasCanceled():
                    nonlocal canceled
                    canceled = True
                    raise StopIteration()
                progress.setValue((i+1)/n*100)
                QtCore.QCoreApplication.processEvents()
            self.animated_action.setChecked(True)
            self.update_chart()
            # plt.rcParams["savefig.dpi"] = 120.0
            if "*.gif" in format_str:
                if not ImageMagickWriter.isAvailable():
                    self.normal_msg.setWindowTitle(self.tr("Error"))
                    self.normal_msg.setText(self.tr("ImageMagick is not installed, please download and install it from its offical website (https://imagemagick.org/index.php)."))
                    self.normal_msg.exec_()
                else:
                    self.__animation.save(filename, writer="imagemagick", fps=30, progress_callback=save_callback)
            elif "*.mp4" in format_str:
                if not FFMpegWriter.isAvailable():
                    self.normal_msg.setWindowTitle(self.tr("Error"))
                    self.normal_msg.setText(self.tr("FFMpeg is not installed, please download and install it from its offical website (https://ffmpeg.org/)."))
                    self.normal_msg.exec_()
                else:
                    self.__animation.save(filename, writer="ffmpeg", fps=30, progress_callback=save_callback)
            # plt.rcParams["savefig.dpi"] = 300.0
            if not canceled:
                progress.setValue(100)

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
