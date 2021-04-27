import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt, QCoreApplication
from PySide2.QtWidgets import QDialog, QGridLayout, QLabel, QComboBox, QCheckBox, QPushButton, QSpinBox, QMessageBox, QFileDialog, QProgressDialog
from QGrain.algorithms.moments import convert_μm_to_φ, convert_φ_to_μm
from QGrain.models.MixedDistributionChartViewModel import MixedDistributionChartViewModel, get_demo_view_model
from QGrain.models.FittingResult import FittingResult
import os

class MixedDistributionChart(QDialog):
    def __init__(self, parent=None, show_mode=True, toolbar=False, use_animation=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Mixed Distribution Chart"))
        self.figure = plt.figure()
        self.axes = self.figure.subplots()
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
        self.interval_label = QLabel(self.tr("Interval [ms]"))
        self.interval_input = QSpinBox()
        self.interval_input.setRange(0, 10000)
        self.interval_input.setValue(30)
        self.interval_input.valueChanged.connect(self.update_animation)
        self.main_layout.addWidget(self.interval_label, 3, 0)
        self.main_layout.addWidget(self.interval_input, 3, 1)
        self.repeat_check_box = QCheckBox(self.tr("Repeat"))
        self.repeat_check_box.setChecked(False)
        self.repeat_check_box.stateChanged.connect(self.update_animation)
        self.save_button = QPushButton(self.tr("Save"))
        self.save_button.clicked.connect(self.save_animation)
        self.main_layout.addWidget(self.repeat_check_box, 4, 0)
        self.main_layout.addWidget(self.save_button, 4, 1)
        self.show_mode = show_mode
        self.animation = None
        self.last_model = None
        self.last_result = None

        if not use_animation:
            self.interval_label.setVisible(False)
            self.interval_input.setVisible(False)
            self.repeat_check_box.setVisible(False)
            self.save_button.setVisible(False)

        self.msg_box = QMessageBox(parent=self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.file_dialog = QFileDialog(parent=self)

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
    def interval(self) -> float:
        return self.interval_input.value()

    @property
    def repeat(self) -> bool:
        return self.repeat_check_box.isChecked()

    def show_demo(self):
        demo_model = get_demo_view_model()
        self.show_model(demo_model)

    def update_chart(self):
        if self.last_model is not None:
            self.show_model(self.last_model)
        elif self.last_result is not None:
            self.show_result(self.last_result)

    def update_animation(self):
        if self.last_result is not None:
            self.show_result(self.last_result)

    def show_model(self, model: MixedDistributionChartViewModel, quick=False):
        if not quick:
            self.last_result = None
            self.last_model = model
            self.interval_label.setEnabled(False)
            self.interval_input.setEnabled(False)
            self.repeat_check_box.setEnabled(False)
            self.save_button.setEnabled(False)

            self.axes.clear()
            x = self.transfer(model.classes_φ)
            if self.xlog:
                self.axes.set_xscale("log")
            self.axes.set_title(model.title)
            self.axes.set_xlabel(self.xlabel)
            self.axes.set_ylabel(self.ylabel)
            self.target = self.axes.plot(x, model.target, c="#ffffff00", marker=".", ms=8, mfc="black", mew=0.0, label=self.tr("Target"))[0]
            # scatter can not be modified from the tool bar
            # self.target = self.axes.scatter(x, model.target, c="black", s=1)
            self.axes.set_xlim(x[0], x[-1])
            self.axes.set_ylim(0.0, round(np.max(model.target)*1.2, 2))
            self.mixed = self.axes.plot(x, model.mixed, c="black", label=self.tr("Mixed"))[0]
            self.components = [self.axes.plot(x, distribution*fraction, c=plt.get_cmap()(i), label=model.component_prefix+str(i+1))[0] for i, (distribution, fraction) in enumerate(zip(model.distributions, model.fractions))]
            if self.show_mode:
                modes = [self.transfer(model.classes_φ[np.unravel_index(np.argmax(distribution), distribution.shape)])  for distribution in model.distributions]
                colors = [plt.get_cmap()(i) for i in range(model.n_components)]
                self.vlines = self.axes.vlines(modes, 0.0, round(np.max(model.target)*1.2, 2), colors=colors)
            self.axes.legend(loc="upper left")
            self.figure.tight_layout()
            self.canvas.draw()
        else:
            self.mixed.set_ydata(model.mixed)
            for comp, distribution, fraction in zip(self.components, model.distributions, model.fractions):
                comp.set_ydata(distribution*fraction)
            if self.show_mode:
                modes = [self.transfer(model.classes_φ[np.unravel_index(np.argmax(distribution), distribution.shape)])  for distribution in model.distributions]
                self.vlines.set_offsets(modes)
            self.canvas.draw()

    def show_result(self, result: FittingResult):
        if self.animation is not None:
            self.animation._stop()
        self.last_model = None
        self.last_result = result
        self.interval_label.setEnabled(True)
        self.interval_input.setEnabled(True)
        self.repeat_check_box.setEnabled(True)
        self.save_button.setEnabled(True)

        models = iter(result.view_models)
        first = next(models)
        x = self.transfer(first.classes_φ)
        self.axes.cla()
        if self.xlog:
            self.axes.set_xscale("log")
        self.axes.set_title(first.title)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.target = self.axes.plot(x, first.target, c="#ffffff00", marker=".", ms=8, mfc="black", mew=0.0)[0]
        self.axes.set_xlim(x[0], x[-1])
        self.axes.set_ylim(0.0, round(np.max(first.target)*1.2, 2))
        self.figure.tight_layout()
        # self.canvas.draw()
        colors = [plt.get_cmap()(i) for i in range(first.n_components)]
        def init():
            model = first
            self.mixed = self.axes.plot(x, model.mixed, c="black")[0]
            self.components = [self.axes.plot(x, distribution*fraction, c=plt.get_cmap()(i))[0] for i, (distribution, fraction) in enumerate(zip(model.distributions, model.fractions))]
            if self.show_mode:
                modes = [self.transfer(model.classes_φ[np.unravel_index(np.argmax(distribution), distribution.shape)])  for distribution in model.distributions]
                self.vlines = self.axes.vlines(modes, 0.0, round(np.max(model.target)*1.2, 2), colors=colors)
            return self.mixed, self.vlines, *self.components
        def animate(current):
            model = current
            self.mixed.set_ydata(current.mixed)
            for line, distribution, fraction in zip(self.components, model.distributions, model.fractions):
                line.set_ydata(distribution*fraction)
            if self.show_mode:
                self.vlines.remove()
                modes = [self.transfer(model.classes_φ[np.unravel_index(np.argmax(distribution), distribution.shape)])  for distribution in model.distributions]
                self.vlines = self.axes.vlines(modes, 0.0, round(np.max(model.target)*1.2, 2), colors=colors)
            return self.mixed, self.vlines, *self.components
        self.animation = FuncAnimation(self.figure, animate, frames=models, init_func=init,
                                       interval=self.interval, blit=True,
                                       repeat=self.repeat, repeat_delay=3.0, save_count=result.n_iterations)

    def save_animation(self):
        if self.last_result is not None:
            if not ImageMagickWriter.isAvailable():
                self.msg_box.setWindowTitle(self.tr("Error"))
                self.msg_box.setText(self.tr("ImageMagick is not installed, please download and install it from its offical website (https://imagemagick.org/index.php)."))
                self.msg_box.exec_()
            else:
                filename, _  = self.file_dialog.getSaveFileName(self, self.tr("Select Filename"),
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
                self.show_result(self.last_result)
                self.animation.save(filename, writer="imagemagick", fps=30, progress_callback=save_callback)
                if not canceled:
                    progress.setValue(100)


if __name__ == "__main__":
    import sys
    from QGrain.entry import setup_app
    app = setup_app()
    canvas = MixedDistributionChart(toolbar=True)
    canvas.show()
    canvas.show_demo()
    sys.exit(app.exec_())
