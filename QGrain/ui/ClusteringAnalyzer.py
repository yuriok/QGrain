__all__ = ["ClusteringAnalyzer"]

import logging

from PySide6 import QtCore, QtWidgets
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

from ..models import Dataset
from ..io import save_clustering
from ..charts.HierarchicalChart import HierarchicalChart

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
        self.main_layout.addWidget(self.chart, 0, 0, 1, 8)
        self.linkage_label = QtWidgets.QLabel(self.tr("Linkage"))
        self.linkage_label.setToolTip(self.tr("The linkage method for calculating the distance between "
                                              "the newly formed cluster and each observation vector."))
        self.linkage_combo_box = QtWidgets.QComboBox()
        self.linkage_combo_box.addItems(linkages)
        self.linkage_combo_box.setCurrentText("ward")
        self.linkage_combo_box.currentTextChanged.connect(lambda name: self.perform())
        self.metric_label = QtWidgets.QLabel(self.tr("Metric"))
        self.metric_label.setToolTip(self.tr("The distance metric."))
        self.metric_combo_box = QtWidgets.QComboBox()
        self.metric_combo_box.addItems(metrics)
        self.metric_combo_box.setCurrentText("euclidean")
        self.metric_combo_box.currentTextChanged.connect(lambda metric: self.perform())
        self.main_layout.addWidget(self.linkage_label, 1, 0)
        self.main_layout.addWidget(self.linkage_combo_box, 1, 1)
        self.main_layout.addWidget(self.metric_label, 1, 2)
        self.main_layout.addWidget(self.metric_combo_box, 1, 3)
        self.p_label = QtWidgets.QLabel(self.tr("p"))
        self.p_label.setToolTip(self.tr("The number of leaves at the bottom level of the figure."))
        self.p_input = QtWidgets.QSpinBox()
        self.p_input.setMinimum(1)
        self.p_input.valueChanged.connect(lambda p: self.perform())
        self.main_layout.addWidget(self.p_label, 1, 4)
        self.main_layout.addWidget(self.p_input, 1, 5)
        self.n_clusters_label = QtWidgets.QLabel(self.tr("Number of Clusters"))
        self.n_clusters_label.setToolTip(self.tr("The number of clusters."))
        self.n_clusters_input = QtWidgets.QSpinBox()
        self.n_clusters_input.setMinimum(2)
        self.main_layout.addWidget(self.n_clusters_label, 1, 6)
        self.main_layout.addWidget(self.n_clusters_input, 1, 7)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self._last_dataset = None
        self._last_result = None

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

    def perform(self):
        if self._last_dataset is None:
            self.logger.warning("The dataset has not been loaded.")
            return
        if self._last_result is not None:
            self.logger.debug("The clustering algorithm has been performed on this dataset, "
                              "use previous result to avoid the repetitive computation.")
            linkage_name, metric_name, linkage_matrix = self._last_result
            if linkage_name == self.linkage_name and metric_name == self.metric_name:
                self.chart.show_matrix(linkage_matrix, p=self.p)
                return
        self.logger.debug(
            f"Calculate the linkage matrix with the method ({self.linkage_name}) and metric ({self.metric_name}).")
        pca = PCA(n_components=0.95)
        transformed = pca.fit_transform(self._last_dataset.distributions)
        try:
            linkage_matrix = linkage(transformed, method=self.linkage_name, metric=self.metric_name)
        except ValueError as e:
            self.logger.error(f"The linkage method {self.linkage_name} is not compatible with "
                              f"the distance metric {self.metric_name}: {e}.")
            self.show_error(self.tr("The linkage method is not compatible with the distance metric."))
            linkage_name, metric_name, linkage_matrix = self._last_result
            # avoid unnecessary chart updating
            self.blockSignals(True)
            self.linkage_combo_box.setCurrentText(linkage_name)
            self.metric_combo_box.setCurrentText(metric_name)
            self.blockSignals(False)
            return
        self._last_result = (self.linkage_name, self.metric_name, linkage_matrix)
        self.chart.show_matrix(linkage_matrix, p=self.p)

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
            linkage_method, metric, linkage_matrix = self._last_result
            flags = fcluster(linkage_matrix, self.n_clusters, criterion="maxclust")
            self.logger.debug(f"Try to save the clustering result. Linkage: {linkage_method}. "
                              f"Metric: {metric}. Number of clusters: {self.n_clusters}.")

            def callback(progress: float):
                if progress_dialog.wasCanceled():
                    raise StopIteration()
                progress_dialog.setValue(int(progress * 100))
                QtCore.QCoreApplication.processEvents()

            save_clustering(self._last_dataset, flags, filename, progress_callback=callback, logger=self.logger)
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

    def retranslate(self):
        self.setWindowTitle(self.tr("Clustering Analyzer"))
        self.linkage_label.setText(self.tr("Linkage"))
        self.linkage_label.setToolTip(self.tr("The linkage method for calculating the distance between "
                                              "the newly formed cluster and each observation vector."))
        self.metric_label.setText(self.tr("Metric"))
        self.metric_label.setToolTip(self.tr("The distance metric."))
        self.p_label.setText(self.tr("p"))
        self.p_label.setToolTip(self.tr("The number of leaves at the bottom level of the figure."))
        self.n_clusters_label.setText(self.tr("Number of Clusters"))
        self.n_clusters_label.setToolTip(self.tr("The number of clusters."))
