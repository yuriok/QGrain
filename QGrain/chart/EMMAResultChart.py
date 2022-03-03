import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter
from PySide6 import QtCore, QtGui, QtWidgets

from ..emma import EMMAResult
from ..statistic import convert_φ_to_μm
from .BaseChart import BaseChart
from .config_matplotlib import normal_color


class EMMAResultChart(BaseChart):
    N_DISPLAY_SAMPLES = 100
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

        self.distance_menu = QtWidgets.QMenu(self.tr("Distance Function")) # type: QtWidgets.QMenu
        self.menu.insertMenu(self.save_figure_action, self.distance_menu)
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

    def show_result(self, result: EMMAResult):
        self.__last_result = result
        self.figure.clear()
        if self.__animation is not None:
            self.__animation._stop()
            self.__animation = None
        classes = self.transfer(result.dataset.classes_φ)
        sample_indexes = np.linspace(1, result.n_samples, result.n_samples)
        iteration_indexes = np.linspace(1, result.n_iterations, result.n_iterations)
        first_result = next(result.history)

        def get_valid(values):
            values = np.array(values)
            return values[~np.isinf(values) & ~np.isnan(values)]
        if self.distance == "cosine":
            min_class_wise_distance = np.max(get_valid(result.get_class_wise_distances(self.distance)))
            max_class_wise_distance = np.min(get_valid(first_result.get_class_wise_distances(self.distance)))
            min_sample_wise_distance = np.max(get_valid(result.get_sample_wise_distances(self.distance)))
            max_sample_wise_distance = np.min(get_valid(first_result.get_sample_wise_distances(self.distance)))
        else:
            min_class_wise_distance = np.min(get_valid(result.get_class_wise_distances(self.distance)))
            max_class_wise_distance = np.max(get_valid(first_result.get_class_wise_distances(self.distance)))
            min_sample_wise_distance = np.min(get_valid(result.get_sample_wise_distances(self.distance)))
            max_sample_wise_distance = np.max(get_valid(first_result.get_sample_wise_distances(self.distance)))
        d_class_wise_distance = max_class_wise_distance - min_class_wise_distance
        min_class_wise_distance -= d_class_wise_distance / 10
        max_class_wise_distance += d_class_wise_distance / 10
        d_sample_wise_distance = max_sample_wise_distance - min_sample_wise_distance
        min_sample_wise_distance -= d_sample_wise_distance / 10
        max_sample_wise_distance += d_sample_wise_distance / 10

        distance_history_axes = self.figure.add_subplot(3, 1, 1)
        distance_history_axes.plot(iteration_indexes, result.get_history_distances(self.distance), c=normal_color())
        distance_history_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        distance_history_axes.set_xlabel("Iteration index")
        distance_history_axes.set_ylabel("Distance")
        distance_history_axes.set_title("Distance history")

        class_wise_distance_axes = self.figure.add_subplot(3, 2, 3)
        if self.xlog:
            class_wise_distance_axes.set_xscale("log")
        class_wise_distance_axes.plot(classes, result.get_class_wise_distances(self.distance), c=normal_color())
        class_wise_distance_axes.set_xlim(classes[0], classes[-1])
        class_wise_distance_axes.set_ylim(min_class_wise_distance, max_class_wise_distance)
        class_wise_distance_axes.set_xlabel(self.xlabel)
        class_wise_distance_axes.set_ylabel("Distance")
        class_wise_distance_axes.set_title("Class-wise")

        sample_wise_distance_axes = self.figure.add_subplot(3, 2, 4)
        sample_wise_distance_axes.plot(sample_indexes, result.get_sample_wise_distances(self.distance), c=normal_color())
        sample_wise_distance_axes.set_xlim(sample_indexes[0], sample_indexes[-1])
        sample_wise_distance_axes.set_ylim(min_sample_wise_distance, max_sample_wise_distance)
        sample_wise_distance_axes.set_xlabel("Sample index")
        sample_wise_distance_axes.set_ylabel("Distance")
        sample_wise_distance_axes.set_title("Sample-wise")

        # get the mode size of each end-members
        modes = [(i, result.dataset.classes_μm[np.unravel_index(np.argmax(result.end_members[i]), result.end_members[i].shape)]) for i in range(result.n_members)]
        # sort them by mode size
        modes.sort(key=lambda x: x[1])
        end_member_axes = self.figure.add_subplot(3, 2, 5)
        if self.xlog:
            end_member_axes.set_xscale("log")
        for i_em, (index, _) in enumerate(modes):
            end_member_axes.plot(classes, result.end_members[index], c=plt.get_cmap()(i_em), label=f"EM{i_em+1}")
        end_member_axes.set_xlim(classes[0], classes[-1])
        end_member_axes.set_ylim(0.0, round(np.max(result.end_members)*1.2, 2))
        end_member_axes.set_xlabel(self.xlabel)
        end_member_axes.set_ylabel(self.ylabel)
        end_member_axes.set_title("Distributions of end members")
        if result.n_members < 6:
            end_member_axes.legend(loc="upper left")

        if result.n_samples > self.N_DISPLAY_SAMPLES:
            interval = result.n_samples // self.N_DISPLAY_SAMPLES
        else:
            interval = 1
        proportion_axes = self.figure.add_subplot(3, 2, 6)
        bottom = np.zeros(result.n_samples)
        for i_em, (index, _) in enumerate(modes):
            proportion_axes.bar(sample_indexes[::interval], result.proportions[:, index][::interval], bottom=bottom[::interval], width=interval, color=plt.get_cmap()(i_em))
            bottom += result.proportions[:, index]
        proportion_axes.set_xlim(sample_indexes[0], sample_indexes[-1])
        proportion_axes.set_ylim(0.0, 1.0)
        proportion_axes.set_xlabel("Sample index")
        proportion_axes.set_ylabel("Proportion")
        proportion_axes.set_title("Proportions of end members")
        self.figure.tight_layout()
        self.canvas.draw()

    def show_animation(self, result: EMMAResult):
        self.__last_result = result
        self.figure.clear()
        if self.__animation is not None:
            self.__animation._stop()
            self.__animation = None

        classes = self.transfer(result.dataset.classes_φ)
        sample_indexes = np.linspace(1, result.n_samples, result.n_samples)
        iteration_indexes = np.linspace(1, result.n_iterations, result.n_iterations)
        distance_series = result.get_history_distances(self.distance)
        min_distance, max_distance = np.min(distance_series), np.max(distance_series)
        first_result = next(result.history)
        def get_valid(values: np.ndarray) -> np.ndarray:
            return values[~np.isinf(values) & ~np.isnan(values)]
        if self.distance == "cosine":
            min_class_wise_distance = np.max(get_valid(result.get_class_wise_distances(self.distance)))
            max_class_wise_distance = np.min(get_valid(first_result.get_class_wise_distances(self.distance)))
            min_sample_wise_distance = np.max(get_valid(result.get_sample_wise_distances(self.distance)))
            max_sample_wise_distance = np.min(get_valid(first_result.get_sample_wise_distances(self.distance)))
        else:
            min_class_wise_distance = np.min(get_valid(result.get_class_wise_distances(self.distance)))
            max_class_wise_distance = np.max(get_valid(first_result.get_class_wise_distances(self.distance)))
            min_sample_wise_distance = np.min(get_valid(result.get_sample_wise_distances(self.distance)))
            max_sample_wise_distance = np.max(get_valid(first_result.get_sample_wise_distances(self.distance)))
        d_class_wise_distance = max_class_wise_distance - min_class_wise_distance
        min_class_wise_distance -= d_class_wise_distance / 10
        max_class_wise_distance += d_class_wise_distance / 10
        d_sample_wise_distance = max_sample_wise_distance - min_sample_wise_distance
        min_sample_wise_distance -= d_sample_wise_distance / 10
        max_sample_wise_distance += d_sample_wise_distance / 10

        self.distance_history_axes = self.figure.add_subplot(3, 1, 1)
        self.distance_history_axes.plot(iteration_indexes, distance_series, c=normal_color())
        self.distance_history_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        self.distance_history_axes.set_xlabel("Iteration index")
        self.distance_history_axes.set_ylabel("Distance")
        self.distance_history_axes.set_title("Distance history")

        self.class_wise_distance_axes = self.figure.add_subplot(3, 2, 3)
        if self.xlog:
            self.class_wise_distance_axes.set_xscale("log")
        self.class_wise_distance_axes.set_xlim(classes[0], classes[-1])
        self.class_wise_distance_axes.set_ylim(min_class_wise_distance, max_class_wise_distance)
        self.class_wise_distance_axes.set_xlabel(self.xlabel)
        self.class_wise_distance_axes.set_ylabel("Distance")
        self.class_wise_distance_axes.set_title("Class-wise")

        self.sample_wise_distance_axes = self.figure.add_subplot(3, 2, 4)
        self.sample_wise_distance_axes.set_xlim(sample_indexes[0], sample_indexes[-1])
        self.sample_wise_distance_axes.set_ylim(min_sample_wise_distance, max_sample_wise_distance)
        self.sample_wise_distance_axes.set_xlabel("Sample index")
        self.sample_wise_distance_axes.set_ylabel("Distance")
        self.sample_wise_distance_axes.set_title("Sample-wise")

        # get the mode size of each end-members
        self.modes = [(i, result.dataset.classes_μm[np.unravel_index(np.argmax(result.end_members[i]), result.end_members[i].shape)]) for i in range(result.n_members)]
        # sort them by mode size
        self.modes.sort(key=lambda x: x[1])
        self.end_member_axes = self.figure.add_subplot(3, 2, 5)
        if self.xlog:
            self.end_member_axes.set_xscale("log")
        self.end_member_axes.set_xlim(classes[0], classes[-1])
        self.end_member_axes.set_ylim(0.0, round(np.max(result.end_members)*1.2, 2))
        self.end_member_axes.set_xlabel(self.xlabel)
        self.end_member_axes.set_ylabel(self.ylabel)
        self.end_member_axes.set_title("Distributions of end members")

        if result.n_samples > self.N_DISPLAY_SAMPLES:
            interval = result.n_samples // self.N_DISPLAY_SAMPLES
        else:
            interval = 1
        self.proportion_axes = self.figure.add_subplot(3, 2, 6)
        self.proportion_axes.set_xlim(sample_indexes[0]-0.5, sample_indexes[-1]-0.5)
        self.proportion_axes.set_ylim(0.0, 1.0)
        self.proportion_axes.set_xlabel("Sample index")
        self.proportion_axes.set_ylabel("Proportion")
        self.proportion_axes.set_title("Proportions of end members")

        # self.figure.tight_layout()
        # self.canvas.draw()
        def init():
            self.iteration_position_line = self.distance_history_axes.plot([1, 1], [min_distance, max_distance], c=normal_color())[0]
            self.class_wise_distance_curve = self.class_wise_distance_axes.plot(classes, result.get_class_wise_distances(self.distance), c=normal_color())[0]
            self.sample_wise_distance_curve = self.sample_wise_distance_axes.plot(sample_indexes, result.get_sample_wise_distances(self.distance), c=normal_color())[0]
            self.end_member_curves = []
            for i_em, (index, _) in enumerate(self.modes):
                end_member_curve = self.end_member_axes.plot(classes, result.end_members[index], c=plt.get_cmap()(i_em), label=f"EM{i_em+1}")[0]
                self.end_member_curves.append(end_member_curve)
            bottom = np.zeros(result.n_samples)
            self.fraction_bars = []
            self.patches = []
            for i_em, (index, _) in enumerate(self.modes):
                bar = self.proportion_axes.bar(sample_indexes[::interval], result.proportions[:, index][::interval], bottom=bottom[::interval], width=interval, color=plt.get_cmap()(i_em))
                self.fraction_bars.append(bar)
                self.patches.extend(bar.patches)
                bottom += result.proportions[:, index]
            return self.iteration_position_line, self.class_wise_distance_curve, self.sample_wise_distance_curve, *(self.end_member_curves + self.patches)

        def animate(args: typing.Tuple[int, EMMAResult]):
            iteration, current = args
            self.iteration_position_line.set_xdata([iteration, iteration])
            self.class_wise_distance_curve.set_ydata(current.get_class_wise_distances(self.distance))
            self.sample_wise_distance_curve.set_ydata(current.get_sample_wise_distances(self.distance))
            for i_em, (index, _) in enumerate(self.modes):
                self.end_member_curves[i_em].set_ydata(current.end_members[index])
            bottom = np.zeros(current.n_samples)
            for i_em, (index, _) in enumerate(self.modes):
                for rect, height, y in zip(self.fraction_bars[i_em].patches, current.proportions[:, index][::interval], bottom[::interval]):
                    rect.set_height(height)
                    rect.set_y(y)
                bottom += current.proportions[:, index]
            return self.iteration_position_line, self.class_wise_distance_curve, self.sample_wise_distance_curve, *(self.end_member_curves + self.patches)

        self.__animation = FuncAnimation(
            self.figure, animate, init_func=init,
            frames=enumerate(result.history), interval=self.animation_interval,
            blit=True, repeat=self.repeat_animation,
            repeat_delay=3.0, save_count=result.n_iterations)

    def save_animation(self):
        if self.__last_result is not None:
            filename, format_str  = self.file_dialog.getSaveFileName(self, self.tr("Save the animation of this EMMA result"), None, self.tr("MPEG-4 Video File (*.mp4);;Graphics Interchange Format (*.gif)"))
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
            self.show_animation(self.__last_result)
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

    def update_chart(self):
        if self.__last_result is not None:
            if self.__animation is not None:
                self.show_animation(self.__last_result)
            else:
                self.show_result(self.__last_result)

    def retranslate(self):
        super().retranslate()
        self.scale_menu.setTitle(self.tr("Scale"))
        for action, (key, name) in zip(self.scale_actions, self.supported_scales):
            action.setText(name)
        self.distance_menu.setTitle(self.tr("Distance Function"))
        for action, (key, name) in zip(self.distance_actions, self.supported_distances):
            action.setText(name)
        self.interval_menu.setTitle(self.tr("Animation Interval"))
        for action, (interval, name) in zip(self.interval_actions, self.supported_intervals):
            action.setText(name)
        self.repeat_action.setText(self.tr("Repeat Animation"))
        self.save_animation_action.setText(self.tr("Save Animation"))
