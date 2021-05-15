import typing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout, QComboBox, QLabel
from QGrain.algorithms.moments import convert_μm_to_φ, convert_φ_to_μm
from QGrain.models.GrainSizeSample import GrainSizeSample


class FrequencyCurve3DChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Frequency Curve 3D Chart"))
        self.figure = plt.figure(figsize=(8, 6))
        self.axes = Axes3D(self.figure, auto_add_to_figure=False)
        self.figure.add_axes(self.axes)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()
        self.supported_scales = [("log", self.tr("Log")),
                                 ("phi", self.tr("φ")),
                                 ("linear", self.tr("Linear"))]
        self.scale_label = QLabel(self.tr("Scale"))
        self.scale_combo_box = QComboBox()
        self.scale_combo_box.addItems([name for key, name in self.supported_scales])
        self.scale_combo_box.setCurrentIndex(1)
        self.scale_combo_box.currentIndexChanged.connect(self.update_chart)
        self.main_layout.addWidget(self.scale_label, 2, 0)
        self.main_layout.addWidget(self.scale_combo_box, 2, 1)

        self.last_samples = []
        self.last_max_frequency = 0.0

    @property
    def scale(self) -> str:
        index = self.scale_combo_box.currentIndex()
        key, name = self.supported_scales[index]
        return key

    @property
    def transfer(self) -> typing.Callable:
        if self.scale == "log":
            return lambda classes_φ: np.log(convert_φ_to_μm(classes_φ))
        elif self.scale == "phi":
            return lambda classes_φ: classes_φ
        elif self.scale == "linear":
            return lambda classes_φ: convert_φ_to_μm(classes_φ)


    @property
    def xlabel(self) -> str:
        if self.scale == "log":
            return self.tr("Ln(grain-size in μm)")
        elif self.scale == "phi":
            return self.tr("Grain-size [φ]")
        elif self.scale == "linear":
            return self.tr("Grain-size [μm]")

    @property
    def ylabel(self) -> str:
        return self.tr("Frequency")

    def update_chart(self):
        self.show_samples(self.last_samples, append=False)

    def show_samples(self, samples: typing.Iterable[GrainSizeSample], append=False):
        append = append and len(self.last_samples) != 0
        if not append:
            self.last_samples = []
        self.axes.clear()
        record_samples = []
        sample_distributions = []
        for sample in self.last_samples:
            record_samples.append(sample)
            sample_distributions.append(sample.distribution)
        for sample in samples:
            record_samples.append(sample)
            sample_distributions.append(sample.distribution)
        Z = np.array(sample_distributions)
        x = self.transfer(record_samples[0].classes_φ)
        y = np.linspace(1, len(Z), len(Z))
        X, Y = np.meshgrid(x, y)
        self.axes.plot_surface(X, Y, Z, rstride=1, cstride=10, cmap="binary")
        self.axes.set_xlim(x[0], x[-1])
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.tr("Sample index"))
        self.axes.set_zlabel(self.ylabel)
        self.last_samples = record_samples
        if self.scale == "linear":
            self.axes.view_init(elev=15.0, azim=45)
        else:
            self.axes.view_init(elev=15.0, azim=-135)
        self.canvas.draw()
