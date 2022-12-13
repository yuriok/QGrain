__all__ = ["ClusteringAnalyzer"]

import logging
import typing

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..statistics import major_statistics
from ..models import Dataset
from ..io import save_clustering
from ..charts.FrequencyGroupChart import FrequencyGroupChart
from ..charts.HierarchicalChart import HierarchicalChart
from ..charts.diagrams import CMDiagramChart


linkages = ("single", "complete", "average", "weighted", "centroid", "median", "ward")
metrics = ("braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming",
           "jaccard", "jensenshannon", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto",
           "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule")


class ClusteringAnalyzer(QtWidgets.QWidget):
    logger = logging.getLogger("QGrain.ClusteringAnalyzer")

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("Clustering Analyzer"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.chart = HierarchicalChart()
        self.main_layout.addWidget(self.chart, 0, 0, 1, 4)
        self.pca_checkbox = QtWidgets.QCheckBox(self.tr("PCA"))
        self.pca_checkbox.setToolTip(self.tr("If use the decomposed features of PCA."))
        self.pca_checkbox.setChecked(True)
        self.pca_ratio_label = QtWidgets.QLabel(self.tr("Variance Ratio"))
        self.pca_ratio_label.setToolTip(self.tr("It determines the numbers of PCs."))
        self.pca_ratio_input = QtWidgets.QDoubleSpinBox()
        self.pca_ratio_input.setDecimals(3)
        self.pca_ratio_input.setRange(0.001, 0.999)
        self.pca_ratio_input.setValue(0.950)
        self.mean_checkbox = QtWidgets.QCheckBox(self.tr("Mean"))
        self.mode_checkbox = QtWidgets.QCheckBox(self.tr("Mode"))
        self.first_percentile_checkbox = QtWidgets.QCheckBox(self.tr("First Percentile (C)"))
        self.median_checkbox = QtWidgets.QCheckBox(self.tr("Median (M)"))
        self.std_checkbox = QtWidgets.QCheckBox(self.tr("Sorting Coefficient"))
        self.skewness_checkbox = QtWidgets.QCheckBox(self.tr("Skewness"))
        self.kurtosis_checkbox = QtWidgets.QCheckBox(self.tr("Kurtosis"))
        self.standardize_checkbox = QtWidgets.QCheckBox(self.tr("Standardize"))
        self.standardize_checkbox.setToolTip(self.tr("Standardization is necessary to balance different features."))
        self.standardize_checkbox.setChecked(True)
        self.main_layout.addWidget(self.pca_checkbox, 1, 0, 1, 2)
        self.main_layout.addWidget(self.pca_ratio_label, 1, 2)
        self.main_layout.addWidget(self.pca_ratio_input, 1, 3)
        self.main_layout.addWidget(self.mean_checkbox, 2, 0)
        self.main_layout.addWidget(self.mode_checkbox, 2, 1)
        self.main_layout.addWidget(self.first_percentile_checkbox, 2, 2)
        self.main_layout.addWidget(self.median_checkbox, 2, 3)
        self.main_layout.addWidget(self.std_checkbox, 3, 0)
        self.main_layout.addWidget(self.skewness_checkbox, 3, 1)
        self.main_layout.addWidget(self.kurtosis_checkbox, 3, 2)
        self.main_layout.addWidget(self.standardize_checkbox, 3, 3)

        self.linkage_label = QtWidgets.QLabel(self.tr("Linkage"))
        self.linkage_label.setToolTip(self.tr("The linkage method for calculating the distance between "
                                              "the newly formed cluster and each observation vector."))
        self.linkage_combo_box = QtWidgets.QComboBox()
        self.linkage_combo_box.addItems(linkages)
        self.linkage_combo_box.setCurrentText("ward")
        self.metric_label = QtWidgets.QLabel(self.tr("Metric"))
        self.metric_label.setToolTip(self.tr("The distance metric."))
        self.metric_combo_box = QtWidgets.QComboBox()
        self.metric_combo_box.addItems(metrics)
        self.metric_combo_box.setCurrentText("euclidean")
        self.main_layout.addWidget(self.linkage_label, 4, 0)
        self.main_layout.addWidget(self.linkage_combo_box, 4, 1)
        self.main_layout.addWidget(self.metric_label, 4, 2)
        self.main_layout.addWidget(self.metric_combo_box, 4, 3)
        self.p_label = QtWidgets.QLabel(self.tr("p"))
        self.p_label.setToolTip(self.tr("The number of leaves at the bottom level of the figure."))
        self.p_input = QtWidgets.QSpinBox()
        self.p_input.setMinimum(1)
        self.main_layout.addWidget(self.p_label, 5, 0)
        self.main_layout.addWidget(self.p_input, 5, 1)
        self.n_clusters_label = QtWidgets.QLabel(self.tr("Number of Clusters"))
        self.n_clusters_label.setToolTip(self.tr("The number of clusters."))
        self.n_clusters_input = QtWidgets.QSpinBox()
        self.n_clusters_input.setMinimum(2)
        self.perform_button = QtWidgets.QPushButton(self.tr("Perform"))
        self.perform_button.clicked.connect(self.perform)
        self.preview_button = QtWidgets.QPushButton(self.tr("Preview"))
        self.preview_button.clicked.connect(self.show_preview)
        self.close_button = QtWidgets.QPushButton(self.tr("Close Preview Charts"))
        self.close_button.clicked.connect(self.close_preview_charts)
        self.show_cm_diagram_button = QtWidgets.QPushButton(self.tr("Show C-M Diagram"))
        self.show_cm_diagram_button.clicked.connect(self.show_cm_diagram)
        self.main_layout.addWidget(self.n_clusters_label, 5, 2)
        self.main_layout.addWidget(self.n_clusters_input, 5, 3)
        self.main_layout.addWidget(self.perform_button, 6, 0)
        self.main_layout.addWidget(self.preview_button, 6, 1)
        self.main_layout.addWidget(self.close_button, 6, 2)
        self.main_layout.addWidget(self.show_cm_diagram_button, 6, 3)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self._last_dataset = None
        self._last_result = None
        self._cluster_group_charts: list[FrequencyGroupChart] = []

    def show_message(self, title: str, message: str):
        self.normal_msg.setWindowTitle(title)
        self.normal_msg.setText(message)
        self.normal_msg.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    @property
    def linkage_name(self) -> str:
        return self.linkage_combo_box.currentText()

    @property
    def metric_name(self) -> str:
        return self.metric_combo_box.currentText()

    @property
    def p(self) -> int:
        return self.p_input.value()

    @property
    def n_clusters(self) -> int:
        return self.n_clusters_input.value()

    @property
    def data_key(self) -> tuple:
        stat_keys: list[str] = []
        if self.mean_checkbox.isChecked():
            stat_keys.append("mean")
        if self.median_checkbox.isChecked():
            stat_keys.append("median")
        if self.mode_checkbox.isChecked():
            stat_keys.append("mode")
        if self.first_percentile_checkbox.isChecked():
            stat_keys.append("first_percentile")
        if self.std_checkbox.isChecked():
            stat_keys.append("std")
        if self.skewness_checkbox.isChecked():
            stat_keys.append("skewness")
        if self.kurtosis_checkbox.isChecked():
            stat_keys.append("kurtosis")
        if self.pca_checkbox.isChecked() and len(stat_keys) == 0:
            if self.standardize_checkbox.isChecked():
                return f"PCA ({self.pca_ratio_input.value()})", "standardized"
            else:
                return f"PCA ({self.pca_ratio_input.value()})",
        elif len(stat_keys) > 0:
            if self.standardize_checkbox.isChecked():
                return f"PCA ({self.pca_ratio_input.value()})", *stat_keys, "standardized"
            else:
                return f"PCA ({self.pca_ratio_input.value()})", *stat_keys
        else:
            return "GSDs",

    def on_dataset_loaded(self, dataset: Dataset):
        if dataset is None:
            return
        self._last_dataset = dataset
        self._last_result = None
        self.p_input.setMaximum(len(dataset))
        self.n_clusters_input.setMaximum(len(dataset) - 1)
        if len(dataset) > 20:
            self.p_input.setValue(10)
        self.perform()

    def _get_data_for_clustering(self) -> typing.Tuple[np.ndarray, typing.Tuple]:
        if self._last_dataset is None:
            self.logger.warning("The dataset has not been loaded.")
            return
        data_key = self.data_key
        stat_keys: list[str] = []
        if self.mean_checkbox.isChecked():
            stat_keys.append("mean")
        if self.median_checkbox.isChecked():
            stat_keys.append("median")
        if self.mode_checkbox.isChecked():
            stat_keys.append("mode")
        if self.first_percentile_checkbox.isChecked():
            stat_keys.append("first_percentile")
        if self.std_checkbox.isChecked():
            stat_keys.append("std")
        if self.skewness_checkbox.isChecked():
            stat_keys.append("skewness")
        if self.kurtosis_checkbox.isChecked():
            stat_keys.append("kurtosis")
        if self.pca_checkbox.isChecked() and len(stat_keys) == 0:
            pca = PCA(n_components=self.pca_ratio_input.value())
            transformed = pca.fit_transform(self._last_dataset.distributions)
            if self.standardize_checkbox.isChecked():
                standardized = StandardScaler().fit_transform(transformed)
                return standardized, data_key
            else:
                return transformed, data_key
        elif len(stat_keys) > 0:
            pca = PCA(n_components=self.pca_ratio_input.value())
            transformed = pca.fit_transform(self._last_dataset.distributions)
            all_stats = []
            for sample in self._last_dataset:
                stats = major_statistics(sample.classes, sample.classes_phi, sample.distribution,
                                         is_geometric=True, is_fw57=False)
                selected_stats = [stats[key] for key in stat_keys]
                all_stats.append(selected_stats)
            all_stats = np.array(all_stats)
            concatenated = np.concatenate([transformed, all_stats], axis=1)
            if self.standardize_checkbox.isChecked():
                standardized = StandardScaler().fit_transform(concatenated)
                return standardized, data_key
            else:
                return concatenated, data_key
        else:
            return self._last_dataset.distributions, data_key

    def perform(self):
        if self._last_dataset is None:
            self.logger.warning("The dataset has not been loaded.")
            return
        if self._last_result is not None:
            data_key, linkage_name, metric_name, linkage_matrix = self._last_result
            if data_key == self.data_key and linkage_name == self.linkage_name and metric_name == self.metric_name:
                self.logger.debug("The clustering algorithm has been performed on this dataset, "
                                  "use previous result to avoid the repetitive computation.")
                self.chart.show_matrix(linkage_matrix, p=self.p)
                return
        self.logger.debug(
            f"Calculate the linkage matrix with the data key ({self.data_key}), method ({self.linkage_name}) and metric ({self.metric_name}).")
        data, data_key = self._get_data_for_clustering()
        try:
            linkage_matrix = linkage(data, method=self.linkage_name, metric=self.metric_name)
            self._last_result = (data_key, self.linkage_name, self.metric_name, linkage_matrix)
            self.chart.show_matrix(linkage_matrix, p=self.p)
        except ValueError as e:
            self.logger.error(f"The linkage method {self.linkage_name} is not compatible with "
                              f"the distance metric {self.metric_name}: {e}.")
            self.show_error(self.tr("The linkage method is not compatible with the distance metric."))
            if self._last_result is not None:
                data_key, linkage_name, metric_name, linkage_matrix = self._last_result
                # avoid unnecessary chart updating
                self.blockSignals(True)
                self.linkage_combo_box.setCurrentText(linkage_name)
                self.metric_combo_box.setCurrentText(metric_name)
                self.blockSignals(False)

    def show_preview(self):
        if self._last_dataset is None:
            self.logger.warning("The dataset has not been loaded.")
            return
        if self._last_result is None:
            self.logger.warning("The clustering has not been performed.")
            return
        self.close_preview_charts()
        data_key, linkage_method, metric, linkage_matrix = self._last_result
        flags = fcluster(linkage_matrix, self.n_clusters, criterion="maxclust")
        self.logger.debug(f"Try to preview the clustering result. Data key: {data_key}. "
                          f"Linkage: {linkage_method}. Metric: {metric}. Number of clusters: {self.n_clusters}.")
        unique_flags = np.unique(flags)
        rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
        x0 = rect.x() + 20
        y0 = rect.y() + 20
        max_w = rect.width()
        max_h = rect.height()
        span = 5
        epoch = 0
        x = x0 + epoch % 10 * 20 + span
        y = y0 + epoch % 10 * 20 + span

        for flag in unique_flags:
            distributions = self._last_dataset.distributions[flags == flag]
            chart = FrequencyGroupChart()
            chart.show_frequency_grop(self._last_dataset.classes_phi, distributions,
                                      title=f"Cluster {flag} (n={len(distributions)})")
            self._cluster_group_charts.append(chart)
            chart.show()
            w = chart.width()
            h = chart.height()
            chart.setGeometry(x, y, w, h)
            QtWidgets.QApplication.processEvents()
            x += w + span
            if x > max_w - w:
                x = x0 + epoch % 10 * 40 + span
                y += h + span
            if y > max_h - h:
                epoch += 1
                x = x0 + epoch % 10 * 40 + span
                y = x0 + epoch % 10 * 40 + span

    def close_preview_charts(self):
        for chart in self._cluster_group_charts:
            chart.setVisible(False)
            chart.close()
        self._cluster_group_charts.clear()

    def show_cm_diagram(self):
        if self._last_dataset is None:
            self.logger.warning("The dataset has not been loaded.")
            return
        if self._last_result is None:
            self.logger.warning("The clustering has not been performed.")
            return
        self.close_preview_charts()
        data_key, linkage_method, metric, linkage_matrix = self._last_result
        flags = fcluster(linkage_matrix, self.n_clusters, criterion="maxclust")
        self.logger.debug(f"Try to show the clusters on a C-M diagram. Data key: {data_key}. "
                          f"Linkage: {linkage_method}. Metric: {metric}. Number of clusters: {self.n_clusters}.")
        unique_flags = np.unique(flags)
        chart = CMDiagramChart()
        marker_size = min(max(2, 5000//len(flags)), 8)
        for i, flag in enumerate(unique_flags):
            key = np.equal(flags, flag)
            samples = [sample for in_cluster, sample in zip(key, self._last_dataset) if in_cluster]
            color = plt.get_cmap("gist_rainbow")(i/len(unique_flags))
            chart.show_samples(samples, append=True, ms=marker_size, mfc=color, mew=0.0)
        chart.show()

    def save_result(self):
        if self._last_dataset is None:
            self.logger.error("The dataset has not been loaded.")
            self.show_error(self.tr("The dataset has not been loaded."))
            return
        if self._last_result is None:
            self.logger.error("The clustering algorithm has not been performed.")
            self.show_error(self.tr("The clustering algorithm has not been performed."))
            return
        filename, _ = self.file_dialog.getSaveFileName(
            self, self.tr("Choose a filename to save the clustering result"), ".", "Microsoft Excel (*.xlsx)")
        if filename is None or filename == "":
            self.logger.info("No filename was selected.")
            return
        progress_dialog = QtWidgets.QProgressDialog(self.tr("Saving the clustering result..."),
                                                    self.tr("Cancel"), 0, 100, self)
        progress_dialog.setWindowTitle("QGrain")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        try:
            data_key, linkage_method, metric, linkage_matrix = self._last_result
            flags = fcluster(linkage_matrix, self.n_clusters, criterion="maxclust")
            self.logger.debug(f"Try to save the clustering result. Data key: {data_key}. "
                              f"Linkage: {linkage_method}. Metric: {metric}. Number of clusters: {self.n_clusters}.")

            def callback(progress: float):
                if progress_dialog.wasCanceled():
                    raise StopIteration()
                progress_dialog.setValue(int(progress * 100))
                QtCore.QCoreApplication.processEvents()

            save_clustering(self._last_dataset, flags, filename, data_key, linkage_method, metric,
                            progress_callback=callback, logger=self.logger)
        except StopIteration:
            self.logger.info("The saving task was canceled.")
        except Exception as e:
            self.logger.exception(f"An unknown exception was raised: {e}. "
                                  f"Please check the logs for more details.", stack_info=True)
            self.show_error(self.tr("An unknown exception was raised. Please check the logs for more details."))
        finally:
            progress_dialog.close()

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.logger.debug("The main window was closed, close all charts of clusters (if have).")
        for chart in self._cluster_group_charts:
            chart.setVisible(False)
            chart.close()
        self._cluster_group_charts.clear()

    def retranslate(self):
        self.setWindowTitle(self.tr("Clustering Analyzer"))
        self.pca_checkbox.setText(self.tr("PCA"))
        self.pca_checkbox.setToolTip(self.tr("If use the decomposed features of PCA."))
        self.pca_ratio_label.setText(self.tr("Variance Ratio"))
        self.pca_ratio_label.setToolTip(self.tr("It determines the numbers of PCs."))
        self.mean_checkbox.setText(self.tr("Mean"))
        self.mode_checkbox.setText(self.tr("Mode"))
        self.first_percentile_checkbox.setText(self.tr("First Percentile (C)"))
        self.median_checkbox.setText(self.tr("Median (M)"))
        self.std_checkbox.setText(self.tr("Sorting Coefficient"))
        self.skewness_checkbox.setText(self.tr("Skewness"))
        self.kurtosis_checkbox.setText(self.tr("Kurtosis"))
        self.standardize_checkbox.setText(self.tr("Standardize"))
        self.standardize_checkbox.setToolTip(self.tr("Standardization is necessary to balance different features."))
        self.linkage_label.setText(self.tr("Linkage"))
        self.linkage_label.setToolTip(self.tr("The linkage method for calculating the distance between "
                                              "the newly formed cluster and each observation vector."))
        self.metric_label.setText(self.tr("Metric"))
        self.metric_label.setToolTip(self.tr("The distance metric."))
        self.p_label.setText(self.tr("p"))
        self.p_label.setToolTip(self.tr("The number of leaves at the bottom level of the figure."))
        self.n_clusters_label.setText(self.tr("Number of Clusters"))
        self.n_clusters_label.setToolTip(self.tr("The number of clusters."))
        self.perform_button.setText(self.tr("Perform"))
        self.preview_button.setText(self.tr("Preview"))
        self.close_button.setText(self.tr("Close Preview Charts"))
        self.show_cm_diagram_button.setText(self.tr("Show C-M Diagram"))
