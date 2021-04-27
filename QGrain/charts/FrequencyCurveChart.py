import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout, QComboBox, QLabel
from QGrain.algorithms.moments import convert_μm_to_φ, convert_φ_to_μm
from QGrain.models.GrainSizeSample import GrainSizeSample


class FrequencyCurveChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Frequency Curve Chart"))
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

        self.last_samples = []
        self.last_max_frequency = 0.0

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

    def update_chart(self):
        self.show_samples(self.last_samples, append=False)

    def show_samples(self, samples: typing.Iterable[GrainSizeSample], append=False, title=None):
        append = append and len(self.last_samples) != 0
        if not append:
            self.axes.clear()
            self.last_samples = []
        max_frequency = self.last_max_frequency
        for i, sample in enumerate(samples):
            self.last_samples.append(sample)
            if i == 0:
                x = self.transfer(sample.classes_φ)
                if not append:
                    if self.xlog:
                        self.axes.set_xscale("log")
                    if title is None:
                        self.axes.set_title(self.tr("Frequency Curves of Samples"))
                    else:
                        self.axes.set_title(title)
                    self.axes.set_xlabel(self.xlabel)
                    self.axes.set_ylabel(self.ylabel)
                    self.axes.set_xlim(x[0], x[-1])
            self.axes.plot(x, sample.distribution, c="black")
            sample_max_freq = np.max(sample.distribution)
            if sample_max_freq > max_frequency:
                max_frequency = sample_max_freq
        self.axes.set_ylim(0.0, round(max_frequency*1.2, 2))
        self.last_max_frequency = max_frequency
        self.figure.tight_layout()
        self.canvas.draw()
