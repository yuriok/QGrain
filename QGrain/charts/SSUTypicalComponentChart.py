import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QGridLayout,
                               QLabel, QPushButton, QSpinBox)
from QGrain.algorithms.moments import convert_μm_to_φ, convert_φ_to_μm
from QGrain.models.GrainSizeSample import GrainSizeSample
from sklearn.cluster import OPTICS


class SSUTypicalComponentChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Frequency Curve Chart"))
        self.figure = plt.figure(figsize=(8, 4))
        self.clustering_axes = self.figure.add_subplot(1, 2, 1)
        self.component_axes = self.figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 4)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 4)
        if not toolbar:
            self.toolbar.hide()
        self.supported_scales = [("log-linear", self.tr("Log-linear")),
                                 ("log", self.tr("Log")),
                                 ("phi", self.tr("φ")),
                                 ("linear", self.tr("Linear"))]
        self.AXIS_LIST = [self.tr("Mean [φ]"), self.tr("STD [φ]"),
                          self.tr("Skewness"), self.tr("Kurtosis")]
        self.x_axis_label = QLabel(self.tr("X Axis"))
        self.x_axis_combo_box = QComboBox()
        self.x_axis_combo_box.addItems(self.AXIS_LIST)
        self.y_axis_label = QLabel(self.tr("Y Axis"))
        self.y_axis_combo_box = QComboBox()
        self.y_axis_combo_box.addItems(self.AXIS_LIST)
        self.y_axis_combo_box.setCurrentIndex(1)
        self.main_layout.addWidget(self.x_axis_label, 2, 0)
        self.main_layout.addWidget(self.x_axis_combo_box, 2, 1)
        self.main_layout.addWidget(self.y_axis_label, 3, 0)
        self.main_layout.addWidget(self.y_axis_combo_box, 3, 1)
        self.scale_label = QLabel(self.tr("Scale"))
        self.scale_combo_box = QComboBox()
        self.scale_combo_box.addItems([name for key, name in self.supported_scales])
        self.main_layout.addWidget(self.scale_label, 2, 2)
        self.main_layout.addWidget(self.scale_combo_box, 2, 3)
        self.min_samples_label = QLabel(self.tr("Minimum Samples"))
        self.min_samples_input = QDoubleSpinBox()
        self.min_samples_input.setRange(0.01, 0.99)
        self.min_samples_input.setValue(0.03)
        self.min_cluster_size_label = QLabel(self.tr("Minimum Cluster Size"))
        self.min_cluster_size_input = QDoubleSpinBox()
        self.min_cluster_size_input.setRange(0.01, 0.99)
        self.min_cluster_size_input.setValue(0.1)
        self.xi_label = QLabel(self.tr("xi"))
        self.xi_input = QDoubleSpinBox()
        self.xi_input.setRange(0.01, 0.99)
        self.xi_input.setValue(0.05)
        self.main_layout.addWidget(self.min_samples_label, 3, 2)
        self.main_layout.addWidget(self.min_samples_input, 3, 3)
        self.main_layout.addWidget(self.min_cluster_size_label, 4, 0)
        self.main_layout.addWidget(self.min_cluster_size_input, 4, 1)
        self.main_layout.addWidget(self.xi_label, 4, 2)
        self.main_layout.addWidget(self.xi_input, 4, 3)
        self.update_chart_button = QPushButton(self.tr("Update Chart"))
        self.update_chart_button.clicked.connect(self.update_chart)
        self.save_typical_button = QPushButton(self.tr("Save Typical"))
        self.save_typical_button.clicked.connect(self.save_typical)
        self.main_layout.addWidget(self.update_chart_button, 5, 0, 1, 2)
        self.main_layout.addWidget(self.save_typical_button, 5, 2, 1, 2)

        self.last_results = None
        self.data_to_clustering = None
        self.stacked_components = None

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
        if self.last_results is None:
            return
        x = self.transfer(self.last_results[0].classes_φ)

        cluster = OPTICS(min_samples=self.min_samples_input.value(),
                         min_cluster_size=self.min_cluster_size_input.value(),
                         xi=self.xi_input.value())
        flags = cluster.fit_predict(self.data_to_clustering)
        cmap = plt.get_cmap()

        self.clustering_axes.clear()
        flag_set = set(flags)
        for flag in flag_set:
            key = np.equal(flags, flag)
            if flag == -1:
                c = "#7a7374"
                label = self.tr("Not Clustered")
            else:
                c = cmap(flag)
                label = self.tr("Cluster{0}").format(flag+1)
            self.clustering_axes.plot(self.data_to_clustering[key][:, self.x_axis_combo_box.currentIndex()],
                                      self.data_to_clustering[key][:, self.y_axis_combo_box.currentIndex()],
                                      c="#ffffff00", marker=".", ms=8, mfc=c, mew=0.0,
                                      zorder=flag,
                                      label=label)
        if len(flag_set) < 6:
            self.clustering_axes.legend(loc="upper left")
        self.clustering_axes.set_xlabel(self.x_axis_combo_box.currentText())
        self.clustering_axes.set_ylabel(self.y_axis_combo_box.currentText())
        self.clustering_axes.set_title(self.tr("Clustering of Components"))

        self.component_axes.clear()
        if self.xlog:
            self.component_axes.set_xscale("log")

        for flag in flag_set:
            if flag == -1:
                c = "#7a7374"
            else:
                c = cmap(flag)
            key = np.equal(flags, flag)
            for distribution in self.stacked_components[key]:
                self.component_axes.plot(x, distribution, c=c, zorder=flag)

            if flag != -1:
                typical = np.mean(self.stacked_components[key], axis=0)
                self.component_axes.plot(x, typical, c="black", zorder=1e10, ls="--", linewidth=1)

        self.component_axes.set_title(self.tr("Typical Components"))
        self.component_axes.set_xlabel(self.xlabel)
        self.component_axes.set_ylabel(self.ylabel)
        self.component_axes.set_xlim(x[0], x[-1])

        self.figure.tight_layout()
        self.canvas.draw()

    def show_typical(self, results: typing.Iterable[GrainSizeSample]):
        if len(results) == 0:
            return
        keys_to_clustering = ["mean", "std", "skewness", "kurtosis"]
        data_to_clustering = []
        stacked_components = []
        for result in results:
            for component in result.components:
                data_to_clustering.append([component.logarithmic_moments[key] for key in keys_to_clustering])
                stacked_components.append(component.distribution)
        # convert to numpy array
        data_to_clustering = np.array(data_to_clustering)
        stacked_components = np.array(stacked_components)
        self.last_results = results
        self.data_to_clustering = data_to_clustering
        self.stacked_components = stacked_components
        self.update_chart()

    def save_typical(self):
        if self.last_results is None:
            return
        # TODO: ADD SUPPORT
