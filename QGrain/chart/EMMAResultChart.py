import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter
from PySide6 import QtCore, QtGui, QtWidgets

from ..emma import EMMAResult
from ..statistics import to_microns
from .BaseChart import BaseChart, get_image_by_proportions
from .config_matplotlib import normal_color


class EMMAResultChart(BaseChart):
    N_DISPLAY_SAMPLES = 200
    def __init__(self, parent=None, figsize=(4, 6)):
        super().__init__(parent=parent, figsize=figsize)
        # self.axes = self.figure.subplots()
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

        self.distance_menu = QtWidgets.QMenu(self.tr("Distance Function")) # type: QtWidgets.QMenu
        self.menu.insertMenu(self.edit_figure_action, self.distance_menu)
        self.distance_group = QtGui.QActionGroup(self.distance_menu)
        self.distance_group.setExclusive(True)
        self.distance_actions = [] # type: list[QtGui.QAction]
        for key, name in self.supported_distances:
            distance_action = self.distance_group.addAction(name) # type: QtGui.QAction
            distance_action.setCheckable(True)
            distance_action.triggered.connect(self.update_chart)
            self.distance_menu.addAction(distance_action)
            self.distance_actions.append(distance_action)
        self.distance_actions[5].setChecked(True)

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

        self.repeat_action = QtGui.QAction(self.tr("Repeat Animation")) # type: QtGui.QAction
        self.repeat_action.triggered.connect(self.update_chart)
        self.menu.insertAction(self.edit_figure_action, self.repeat_action)
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
    def supported_distances(self) -> typing.List[typing.Tuple[str, str]]:
        distances = [
            ("1-norm", self.tr("1 Norm")),
            ("2-norm", self.tr("2 Norm")),
            ("3-norm", self.tr("3 Norm")),
            ("4-norm", self.tr("4 Norm")),
            ("MSE", self.tr("Mean Squared Error")),
            ("log10MSE", self.tr("Logarithmic Mean Squared Error")),
            ("cosine", self.tr("Cosine Error")),
            ("angular", self.tr("Angular Error"))]
        return distances

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
    def distance(self) -> str:
        for i, distance_action in enumerate(self.distance_actions):
            if distance_action.isChecked():
                key, name = self.supported_distances[i]
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
            return lambda classes_φ: to_microns(classes_φ)
        elif self.scale == "log":
            return lambda classes_φ: np.log(to_microns(classes_φ))
        elif self.scale == "phi":
            return lambda classes_φ: classes_φ
        elif self.scale == "linear":
            return lambda classes_φ: to_microns(classes_φ)

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
        self.edit_figure_action.setEnabled(self.__last_result is not None and not self.animated)
        self.save_figure_action.setEnabled(self.__last_result is not None and not self.animated)
        self.save_animation_action.setEnabled(self.__last_result is not None and self.animated)
        self.menu.popup(QtGui.QCursor.pos())

    def show_chart(self, result: EMMAResult):
        classes = self.transfer(result.dataset.classes_phi)
        sample_indexes = np.linspace(1, result.n_samples, result.n_samples)
        iteration_indexes = np.linspace(1, result.n_iterations, result.n_iterations)
        interval = max(1, result.n_samples // self.N_DISPLAY_SAMPLES)
        GSDs = result.dataset.distributions

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

        distance_axes = self.figure.add_subplot(2, 2, 2)
        distance_axes.plot(iteration_indexes, result.get_distance_series(self.distance), c=normal_color())
        distance_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        distance_axes.set_xlabel("Iteration")
        distance_axes.set_ylabel("Distance")
        distance_axes.set_title("Distance variation")

        end_member_axes = self.figure.add_subplot(2, 2, 3)
        if self.xlog:
            end_member_axes.set_xscale("log")
        for i in range(result.n_members):
            end_member_axes.plot(classes, result.end_members[i], c=plt.get_cmap()(i), label=f"EM{i+1}", zorder=10+i)
        end_member_axes.set_xlim(classes[0], classes[-1])
        end_member_axes.set_ylim(0.0, round(np.max(result.end_members)*1.2, 2))
        end_member_axes.set_xlabel(self.xlabel)
        end_member_axes.set_ylabel(self.ylabel)
        end_member_axes.set_title("End members")

        proportion_axes = self.figure.add_subplot(2, 2, 4)
        image = get_image_by_proportions(result.proportions, resolution=100)
        proportion_axes.imshow(image, plt.get_cmap(), aspect="auto", vmin=0, vmax=9)
        proportion_axes.set_xlim(0, result.n_samples-1)
        proportion_axes.set_ylim(0.0, 100.0)
        proportion_axes.set_xlabel("Sample index")
        proportion_axes.set_ylabel("Proportion [%]")
        proportion_axes.set_title("Proportions")

        self.figure.tight_layout()
        self.canvas.draw()

    def show_animation(self, result: EMMAResult):
        classes = self.transfer(result.dataset.classes_phi)
        sample_indexes = np.linspace(1, result.n_samples, result.n_samples)
        iteration_indexes = np.linspace(1, result.n_iterations, result.n_iterations)
        distance_series = result.get_distance_series(self.distance)
        min_distance, max_distance = np.min(distance_series), np.max(distance_series)
        interval = max(1, result.n_samples // self.N_DISPLAY_SAMPLES)
        GSDs = result.dataset.distributions

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

        distance_axes = self.figure.add_subplot(2, 2, 2)
        distance_axes.plot(iteration_indexes, result.get_distance_series(self.distance), c=normal_color())
        distance_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        distance_axes.set_xlabel("Iteration")
        distance_axes.set_ylabel("Distance")
        distance_axes.set_title("Distance variation")

        end_member_axes = self.figure.add_subplot(2, 2, 3)
        if self.xlog:
            end_member_axes.set_xscale("log")
        end_member_axes.set_xlim(classes[0], classes[-1])
        end_member_axes.set_ylim(0.0, round(np.max(result.end_members)*1.2, 2))
        end_member_axes.set_xlabel(self.xlabel)
        end_member_axes.set_ylabel(self.ylabel)
        end_member_axes.set_title("End members")

        proportion_axes = self.figure.add_subplot(2, 2, 4)
        proportion_axes.set_xlim(0, result.n_samples-1)
        proportion_axes.set_ylim(0.0, 100.0)
        proportion_axes.set_xlabel("Sample index")
        proportion_axes.set_ylabel("Proportion [%]")
        proportion_axes.set_title("Proportions")

        # self.figure.tight_layout()
        # self.canvas.draw()
        def init():
            self.iteration_line = distance_axes.plot([1, 1], [min_distance, max_distance], c=normal_color())[0]
            self.end_member_curves = []
            for i in range(result.n_members):
                end_member_curve = end_member_axes.plot(classes, result.end_members[i], c=plt.get_cmap()(i), label=f"EM{i+1}")[0]
                self.end_member_curves.append(end_member_curve)
            image = get_image_by_proportions(result.proportions, resolution=100)
            self.proportion_image = proportion_axes.imshow(image, plt.get_cmap(), aspect="auto", vmin=0, vmax=9)
            return self.iteration_line, self.proportion_image, *self.end_member_curves

        def animate(args: typing.Tuple[int, EMMAResult]):
            iteration, current = args
            self.iteration_line.set_xdata([iteration, iteration])
            for i in range(current.n_members):
                self.end_member_curves[i].set_ydata(current.end_members[i])
            image = get_image_by_proportions(current.proportions, resolution=100)
            self.proportion_image.set_data(image)
            return self.iteration_line, self.proportion_image, *self.end_member_curves

        self.__animation = FuncAnimation(
            self.figure, animate, init_func=init,
            frames=enumerate(result.history), interval=self.animation_interval,
            blit=True, repeat=self.repeat_animation,
            repeat_delay=3.0, save_count=result.n_iterations)

    def show_result(self, result: EMMAResult):
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
                self, self.tr("Choose a filename to save the animation of this EMMA result"),
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
        self.distance_menu.setTitle(self.tr("Distance Function"))
        for action, (key, name) in zip(self.distance_actions, self.supported_distances):
            action.setText(name)
        self.animated_action.setText(self.tr("Animated"))
        self.interval_menu.setTitle(self.tr("Animation Interval"))
        for action, (interval, name) in zip(self.interval_actions, self.supported_intervals):
            action.setText(name)
        self.repeat_action.setText(self.tr("Repeat Animation"))
        self.save_animation_action.setText(self.tr("Save Animation"))
