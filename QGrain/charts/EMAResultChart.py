import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import QCoreApplication, Qt
from PySide2.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                               QGridLayout, QLabel, QMessageBox,
                               QProgressDialog, QPushButton, QSpinBox)
from QGrain.algorithms.moments import convert_μm_to_φ, convert_φ_to_μm
from QGrain.models.EMAResult import EMAResult


class EMAResultChart(QDialog):
    N_DISPLAY_SAMPLES = 200
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("EMA Result Chart"))
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()
        self.supported_scales = [("log-linear", self.tr("Log-linear")),
                                 ("log", self.tr("Log")),
                                 ("phi", self.tr("φ")),
                                 ("linear", self.tr("Linear"))]
        self.scale_label = QLabel(self.tr("Scale"))
        self.scale_combo_box = QComboBox()
        self.scale_combo_box.addItems([name for key, name in self.supported_scales])
        self.scale_combo_box.currentIndexChanged.connect(self.update_chart)
        self.main_layout.addWidget(self.scale_label, 2, 0)
        self.main_layout.addWidget(self.scale_combo_box, 2, 1)

        self.supported_distances = ("1-norm", "2-norm", "3-norm", "4-norm", "MSE", "log10MSE", "cosine", "angular")
        self.distance_label = QLabel(self.tr("Distance"))
        self.distance_combo_box = QComboBox()
        self.distance_combo_box.addItems(self.supported_distances)
        self.distance_combo_box.setCurrentText("log10MSE")
        self.distance_combo_box.currentIndexChanged.connect(self.update_chart)
        self.main_layout.addWidget(self.distance_label, 3, 0)
        self.main_layout.addWidget(self.distance_combo_box, 3, 1)
        self.animated_checkbox = QCheckBox(self.tr("Animated"))
        self.animated_checkbox.setChecked(False)
        self.animated_checkbox.stateChanged.connect(self.on_animated_changed)

        self.main_layout.addWidget(self.animated_checkbox, 4, 0)
        self.interval_label = QLabel(self.tr("Interval [ms]"))
        self.interval_input = QSpinBox()
        self.interval_input.setRange(0, 10000)
        self.interval_input.setValue(30)
        self.interval_input.valueChanged.connect(self.update_chart)
        self.main_layout.addWidget(self.interval_label, 5, 0)
        self.main_layout.addWidget(self.interval_input, 5, 1)
        self.repeat_check_box = QCheckBox(self.tr("Repeat"))
        self.repeat_check_box.setChecked(False)
        self.repeat_check_box.stateChanged.connect(self.update_chart)
        self.save_button = QPushButton(self.tr("Save"))
        self.save_button.clicked.connect(self.save_animation)
        self.main_layout.addWidget(self.repeat_check_box, 6, 0)
        self.main_layout.addWidget(self.save_button, 6, 1)

        self.animation = None
        self.last_result = None

        self.msg_box = QMessageBox(parent=self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.file_dialog = QFileDialog(parent=self)
        self.on_animated_changed()

    @property
    def scale(self) -> str:
        index = self.scale_combo_box.currentIndex()
        key, name = self.supported_scales[index]
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
            return self.tr("Grain size [μm]")
        elif self.scale == "log":
            return self.tr("Ln(grain size in μm)")
        elif self.scale == "phi":
            return self.tr("Grain size [φ]")
        elif self.scale == "linear":
            return self.tr("Grain size [μm]")

    @property
    def ylabel(self) -> str:
        return self.tr("Frequency")

    @property
    def xlog(self) -> bool:
        if self.scale == "log-linear":
            return True
        else:
            return False

    @property
    def distance(self) -> str:
        return self.distance_combo_box.currentText()

    @property
    def interval(self) -> float:
        return self.interval_input.value()

    @property
    def repeat(self) -> bool:
        return self.repeat_check_box.isChecked()

    def update_chart(self):
        if self.last_result is not None:
            self.show_result(self.last_result)

    def on_animated_changed(self):
        if self.animated_checkbox.isChecked():
            enabled = True
        else:
            enabled = False
        self.interval_label.setEnabled(enabled)
        self.interval_input.setEnabled(enabled)
        self.repeat_check_box.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.update_chart()

    def show_final_result(self, result: EMAResult):
        self.last_result = result
        self.figure.clear()
        if self.animation is not None:
            self.animation._stop()
            self.animation = None
        classes = self.transfer(result.dataset.classes_φ)
        sample_indexes = np.linspace(1, result.n_samples, result.n_samples)
        iteration_indexes = np.linspace(1, result.n_iterations, result.n_iterations)
        first_result = next(result.history)
        def get_valid(values):
            values = np.array(values)
            return values[~np.isinf(values) & ~np.isnan(values)]
        if self.distance == "cosine":
            min_class_wise_distance = np.max(get_valid(result.get_class_wise_distance_series(self.distance)))
            max_class_wise_distance = np.min(get_valid(first_result.get_class_wise_distance_series(self.distance)))
            min_sample_wise_distance = np.max(get_valid(result.get_sample_wise_distance_series(self.distance)))
            max_sample_wise_distance = np.min(get_valid(first_result.get_sample_wise_distance_series(self.distance)))
        else:
            min_class_wise_distance = np.min(get_valid(result.get_class_wise_distance_series(self.distance)))
            max_class_wise_distance = np.max(get_valid(first_result.get_class_wise_distance_series(self.distance)))
            min_sample_wise_distance = np.min(get_valid(result.get_sample_wise_distance_series(self.distance)))
            max_sample_wise_distance = np.max(get_valid(first_result.get_sample_wise_distance_series(self.distance)))
        d_class_wise_distance = max_class_wise_distance - min_class_wise_distance
        min_class_wise_distance -= d_class_wise_distance / 10
        max_class_wise_distance += d_class_wise_distance / 10
        d_sample_wise_distance = max_sample_wise_distance - min_sample_wise_distance
        min_sample_wise_distance -= d_sample_wise_distance / 10
        max_sample_wise_distance += d_sample_wise_distance / 10

        distance_history_axes = self.figure.add_subplot(3, 1, 1)
        distance_history_axes.plot(iteration_indexes, result.get_distance_series(self.distance))
        distance_history_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        distance_history_axes.set_xlabel(self.tr("Iteration Index"))
        distance_history_axes.set_ylabel(self.tr("Distance"))
        distance_history_axes.set_title(self.tr("Distance History"))

        class_wise_distance_axes = self.figure.add_subplot(3, 2, 3)
        if self.xlog:
            class_wise_distance_axes.set_xscale("log")
        class_wise_distance_axes.plot(classes, result.get_class_wise_distance_series(self.distance))
        class_wise_distance_axes.set_xlim(classes[0], classes[-1])
        class_wise_distance_axes.set_ylim(min_class_wise_distance, max_class_wise_distance)
        class_wise_distance_axes.set_xlabel(self.xlabel)
        class_wise_distance_axes.set_ylabel(self.tr("Distance"))
        class_wise_distance_axes.set_title(self.tr("Class-wise Distances"))

        sample_wise_distance_axes = self.figure.add_subplot(3, 2, 4)
        sample_wise_distance_axes.plot(sample_indexes, result.get_sample_wise_distance_series(self.distance))
        sample_wise_distance_axes.set_xlim(sample_indexes[0], sample_indexes[-1])
        sample_wise_distance_axes.set_ylim(min_sample_wise_distance, max_sample_wise_distance)
        sample_wise_distance_axes.set_xlabel(self.tr("Sample Index"))
        sample_wise_distance_axes.set_ylabel(self.tr("Distance"))
        sample_wise_distance_axes.set_title(self.tr("Sample-wise Distances"))

        # get the mode size of each end-members
        modes = [(i, result.dataset.classes_μm[np.unravel_index(np.argmax(result.end_members[i]), result.end_members[i].shape)]) for i in range(result.n_members)]
        # sort them by mode size
        modes.sort(key=lambda x: x[1])
        end_member_axes = self.figure.add_subplot(3, 2, 5)
        if self.xlog:
            end_member_axes.set_xscale("log")
        for i_em, (index, _) in enumerate(modes):
            end_member_axes.plot(classes, result.end_members[index], c=plt.get_cmap()(i_em), label=self.tr("EM{0}").format(i_em+1))
        end_member_axes.set_xlim(classes[0], classes[-1])
        end_member_axes.set_ylim(0.0, round(np.max(result.end_members)*1.2, 2))
        end_member_axes.set_xlabel(self.xlabel)
        end_member_axes.set_ylabel(self.ylabel)
        end_member_axes.set_title(self.tr("End-members"))
        if result.n_members < 6:
            end_member_axes.legend(loc="upper left")

        if result.n_samples > self.N_DISPLAY_SAMPLES:
            interval = result.n_samples // self.N_DISPLAY_SAMPLES
        else:
            interval = 1
        fraction_axes = self.figure.add_subplot(3, 2, 6)
        bottom = np.zeros(result.n_samples)
        for i_em, (index, _) in enumerate(modes):
            fraction_axes.bar(sample_indexes[::interval], result.fractions[:, index][::interval], bottom=bottom[::interval], width=interval, color=plt.get_cmap()(i_em))
            bottom += result.fractions[:, index]
        fraction_axes.set_xlim(sample_indexes[0], sample_indexes[-1])
        fraction_axes.set_ylim(0.0, 1.0)
        fraction_axes.set_xlabel(self.tr("Sample Index"))
        fraction_axes.set_ylabel(self.tr("Fraction"))
        fraction_axes.set_title(self.tr("Sample Fractions"))
        self.figure.tight_layout()
        self.canvas.draw()

    def show_history_animation(self, result: EMAResult):
        self.last_result = result
        self.figure.clear()
        if self.animation is not None:
            self.animation._stop()
            self.animation = None

        classes = self.transfer(result.dataset.classes_φ)
        sample_indexes = np.linspace(1, result.n_samples, result.n_samples)
        iteration_indexes = np.linspace(1, result.n_iterations, result.n_iterations)
        distance_series = result.get_distance_series(self.distance)
        min_distance, max_distance = np.min(distance_series), np.max(distance_series)
        first_result = next(result.history)
        def get_valid(values):
            values = np.array(values)
            return values[~np.isinf(values) & ~np.isnan(values)]
        if self.distance == "cosine":
            min_class_wise_distance = np.max(get_valid(result.get_class_wise_distance_series(self.distance)))
            max_class_wise_distance = np.min(get_valid(first_result.get_class_wise_distance_series(self.distance)))
            min_sample_wise_distance = np.max(get_valid(result.get_sample_wise_distance_series(self.distance)))
            max_sample_wise_distance = np.min(get_valid(first_result.get_sample_wise_distance_series(self.distance)))
        else:
            min_class_wise_distance = np.min(get_valid(result.get_class_wise_distance_series(self.distance)))
            max_class_wise_distance = np.max(get_valid(first_result.get_class_wise_distance_series(self.distance)))
            min_sample_wise_distance = np.min(get_valid(result.get_sample_wise_distance_series(self.distance)))
            max_sample_wise_distance = np.max(get_valid(first_result.get_sample_wise_distance_series(self.distance)))
        d_class_wise_distance = max_class_wise_distance - min_class_wise_distance
        min_class_wise_distance -= d_class_wise_distance / 10
        max_class_wise_distance += d_class_wise_distance / 10
        d_sample_wise_distance = max_sample_wise_distance - min_sample_wise_distance
        min_sample_wise_distance -= d_sample_wise_distance / 10
        max_sample_wise_distance += d_sample_wise_distance / 10

        self.distance_history_axes = self.figure.add_subplot(3, 1, 1)
        self.distance_history_axes.plot(iteration_indexes, distance_series)
        self.distance_history_axes.set_xlim(iteration_indexes[0], iteration_indexes[-1])
        self.distance_history_axes.set_xlabel(self.tr("Iteration Index"))
        self.distance_history_axes.set_ylabel(self.tr("Distance"))
        self.distance_history_axes.set_title(self.tr("Distance History"))

        self.class_wise_distance_axes = self.figure.add_subplot(3, 2, 3)
        if self.xlog:
            self.class_wise_distance_axes.set_xscale("log")
        self.class_wise_distance_axes.set_xlim(classes[0], classes[-1])
        self.class_wise_distance_axes.set_ylim(min_class_wise_distance, max_class_wise_distance)
        self.class_wise_distance_axes.set_xlabel(self.xlabel)
        self.class_wise_distance_axes.set_ylabel(self.tr("Distance"))
        self.class_wise_distance_axes.set_title(self.tr("Class-wise Distances"))

        self.sample_wise_distance_axes = self.figure.add_subplot(3, 2, 4)
        self.sample_wise_distance_axes.set_xlim(sample_indexes[0], sample_indexes[-1])
        self.sample_wise_distance_axes.set_ylim(min_sample_wise_distance, max_sample_wise_distance)
        self.sample_wise_distance_axes.set_xlabel(self.tr("Sample Index"))
        self.sample_wise_distance_axes.set_ylabel(self.tr("Distance"))
        self.sample_wise_distance_axes.set_title(self.tr("Sample-wise Distances"))

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
        self.end_member_axes.set_title(self.tr("End-members"))


        if result.n_samples > self.N_DISPLAY_SAMPLES:
            interval = result.n_samples // self.N_DISPLAY_SAMPLES
        else:
            interval = 1
        self.fraction_axes = self.figure.add_subplot(3, 2, 6)
        self.fraction_axes.set_xlim(sample_indexes[0]-0.5, sample_indexes[-1]-0.5)
        self.fraction_axes.set_ylim(0.0, 1.0)
        self.fraction_axes.set_xlabel(self.tr("Sample Index"))
        self.fraction_axes.set_ylabel(self.tr("Fraction"))
        self.fraction_axes.set_title(self.tr("Sample Fractions"))

        self.figure.tight_layout()
        # self.canvas.draw()
        def init():
            self.iteration_position_line = self.distance_history_axes.plot([1, 1], [min_distance, max_distance], c="black")[0]
            self.class_wise_distance_curve = self.class_wise_distance_axes.plot(classes, result.get_class_wise_distance_series(self.distance), c=plt.get_cmap()(0))[0]
            self.sample_wise_distance_curve = self.sample_wise_distance_axes.plot(sample_indexes, result.get_sample_wise_distance_series(self.distance), c=plt.get_cmap()(0))[0]
            self.end_member_curves = []
            for i_em, (index, _) in enumerate(self.modes):
                end_member_curve = self.end_member_axes.plot(classes, result.end_members[index], c=plt.get_cmap()(i_em), label=self.tr("EM{0}").format(i_em+1))[0]
                self.end_member_curves.append(end_member_curve)
            bottom = np.zeros(result.n_samples)
            self.fraction_bars = []
            self.patches = []
            for i_em, (index, _) in enumerate(self.modes):
                bar = self.fraction_axes.bar(sample_indexes[::interval], result.fractions[:, index][::interval], bottom=bottom[::interval], width=interval, color=plt.get_cmap()(i_em))
                self.fraction_bars.append(bar)
                self.patches.extend(bar.patches)
                bottom += result.fractions[:, index]
            return self.iteration_position_line, self.class_wise_distance_curve, self.sample_wise_distance_curve, *self.end_member_curves, *self.patches

        def animate(args):
            iteration, current = args
            self.iteration_position_line.set_xdata([iteration, iteration])
            self.class_wise_distance_curve.set_ydata(current.get_class_wise_distance_series(self.distance))
            self.sample_wise_distance_curve.set_ydata(current.get_sample_wise_distance_series(self.distance))
            for i_em, (index, _) in enumerate(self.modes):
                self.end_member_curves[i_em].set_ydata(current.end_members[index])
            bottom = np.zeros(current.n_samples)
            for i_em, (index, _) in enumerate(self.modes):
                for rect, height, y in zip(self.fraction_bars[i_em].patches, current.fractions[:, index][::interval], bottom[::interval]):
                    rect.set_height(height)
                    rect.set_y(y)
                bottom += current.fractions[:, index]
            return self.iteration_position_line, self.class_wise_distance_curve, self.sample_wise_distance_curve, *self.end_member_curves, *self.patches

        self.animation = FuncAnimation(self.figure, animate, init_func=init, frames=enumerate(result.history) , interval=self.interval, blit=True, repeat=self.repeat, repeat_delay=3.0, save_count=result.n_iterations)

    def show_result(self, result: EMAResult):
        if self.animated_checkbox.isChecked():
            self.show_history_animation(result)
        else:
            self.show_final_result(result)

    def save_animation(self):
        if self.last_result is not None:
            if not ImageMagickWriter.isAvailable():
                self.msg_box.setWindowTitle(self.tr("Error"))
                self.msg_box.setText(self.tr("ImageMagick is not installed, please download and install it from its offical website (https://imagemagick.org/index.php)."))
                self.msg_box.exec_()
            else:
                filename, _  = self.file_dialog.getSaveFileName(self, self.tr("Save the animation of EMA result"),
                                            None, self.tr("Graphics Interchange Format (*.gif)"))
                if filename is None or filename == "":
                    return
                progress = QProgressDialog(self)
                progress.setRange(0, 100)
                progress.setLabelText(self.tr("Saving Animation [{0} Frames]").format(self.last_result.n_iterations))
                canceled = False
                def save_callback(i, n):
                    if progress.wasCanceled():
                        canceled = True
                        raise StopIteration()
                    progress.setValue((i+1)/n*100)
                    QCoreApplication.processEvents()
                self.show_history_animation(self.last_result)
                self.animation.save(filename, writer="imagemagick", fps=30, progress_callback=save_callback)
                if not canceled:
                    progress.setValue(100)
