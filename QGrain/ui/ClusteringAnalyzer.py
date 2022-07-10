__all__ = ["ClusteringAnalyzer"]

import logging

from PySide6 import QtCore, QtWidgets
from scipy.cluster.hierarchy import fcluster, linkage

from ..chart.HierarchicalChart import HierarchicalChart
from ..io import save_clustering
from ..models import GrainSizeDataset


class ClusteringAnalyzer(QtWidgets.QWidget):
    logger = logging.getLogger("QGrain.ClusteringAnalyzer")
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_ui()
        self.file_dialog = QtWidgets.QFileDialog(self)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.__dataset = None
        self.__distribution_matrix = None
        self.__last_result = None

    def init_ui(self):
        self.setWindowTitle(self.tr("Clustering Analyzer"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.chart = HierarchicalChart()
        self.main_layout.addWidget(self.chart, 0, 0, 1, 8)

        self.supported_methods = [
            "single", "complete",
            "average", "weighted",
            "centroid", "median",
            "ward"]
        self.supported_distances = [
            "braycurtis", "canberra",
            "chebyshev", "cityblock",
            "correlation", "cosine",
            "dice", "euclidean",
            "hamming", "jaccard",
            "jensenshannon", "kulsinski",
            "mahalanobis", "matching",
            "minkowski", "rogerstanimoto",
            "russellrao", "seuclidean",
            "sokalmichener", "sokalsneath",
            "sqeuclidean", "yule"]

        self.linkage_method_label = QtWidgets.QLabel(self.tr("Linkage Method"))
        self.linkage_method_label.setToolTip(self.tr("The linkage method for calculating the distance between the newly formed cluster and each observation vector."))
        self.linkage_method_combo_box = QtWidgets.QComboBox()
        self.linkage_method_combo_box.addItems(self.supported_methods)
        self.linkage_method_combo_box.setCurrentText("ward")
        self.linkage_method_combo_box.currentTextChanged.connect(lambda linkage: self.perform())
        self.distance_label = QtWidgets.QLabel(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The distance metric."))
        self.distance_combo_box = QtWidgets.QComboBox()
        self.distance_combo_box.addItems(self.supported_distances)
        self.distance_combo_box.setCurrentText("euclidean")
        self.distance_combo_box.currentTextChanged.connect(lambda distance: self.perform())
        self.main_layout.addWidget(self.linkage_method_label, 1, 0)
        self.main_layout.addWidget(self.linkage_method_combo_box, 1, 1)
        self.main_layout.addWidget(self.distance_label, 1, 2)
        self.main_layout.addWidget(self.distance_combo_box, 1, 3)

        self.p_label = QtWidgets.QLabel(self.tr("p"))
        self.p_label.setToolTip(self.tr("Controls the number of leaves at the bottom level of the figure."))
        self.p_input = QtWidgets.QSpinBox()
        self.p_input.setMinimum(1)
        self.p_input.valueChanged.connect(lambda p: self.perform())
        self.main_layout.addWidget(self.p_label, 1, 4)
        self.main_layout.addWidget(self.p_input, 1, 5)
        self.n_clusers_label = QtWidgets.QLabel(self.tr("Number of Clusters"))
        self.n_clusers_label.setToolTip(self.tr("Controls the number of clusters of this clustering algorithm."))
        self.n_clusers_input = QtWidgets.QSpinBox()
        self.n_clusers_input.setMinimum(2)
        self.main_layout.addWidget(self.n_clusers_label, 1, 6)
        self.main_layout.addWidget(self.n_clusers_input, 1, 7)

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
    def linkage_method(self) -> str:
        return self.linkage_method_combo_box.currentText()

    @property
    def distance(self) -> str:
        return self.distance_combo_box.currentText()

    @property
    def p(self) -> int:
        return self.p_input.value()

    @property
    def n_clusters(self) -> int:
        return self.n_clusers_input.value()

    def on_dataset_loaded(self, dataset: GrainSizeDataset):
        if dataset is None or not dataset.has_sample:
            return
        self.__dataset = dataset
        self.__distribution_matrix = dataset.distribution_matrix
        self.__last_result = None

        self.p_input.setMaximum(dataset.n_samples)
        self.n_clusers_input.setMaximum(dataset.n_samples-1)
        if dataset.n_samples > 20:
            self.p_input.setValue(10)

        self.perform()

    def perform(self):
        try:
            if self.__last_result is not None:
                self.logger.debug("The clustering algorithm has been performed on this dataset, use previous result to avoid the repetitive computation.")
                linkage_method, distance, linkage_matrix = self.__last_result
                if linkage_method == self.linkage_method and distance == self.distance:
                    self.chart.show_result(linkage_matrix, p=self.p)
                    return
            self.logger.debug(f"Calculate the linkage matrix with the method ({self.linkage_method}) and metric ({self.distance}).")
            linkage_matrix = linkage(self.__distribution_matrix, method=self.linkage_method, metric=self.distance)
            self.__last_result = (self.linkage_method, self.distance, linkage_matrix)
            self.chart.show_result(linkage_matrix, p=self.p)
        except ValueError as e:
            self.logger.error("The linkage method is not compatible with the distance metric.")
            self.show_error(self.tr("The linkage method is not compatible with the distance metric."))
            return

    def save_result(self):
        if self.__last_result is None:
            self.logger.error("The clustering algorithm has not been performed.")
            self.show_error(self.tr("The clustering algorithm has not been performed."))
            return
        filename, _ = self.file_dialog.getSaveFileName(
            None, self.tr("Choose a filename to save the clustering result"),
            None, "Microsoft Excel (*.xlsx)")
        if filename is None or filename == "":
            return
        try:
            linkage_method, distance, linkage_matrix = self.__last_result
            flags = fcluster(linkage_matrix, self.n_clusters, criterion="maxclust")
            self.logger.debug(f"Try to save the clustering result. Linkage method: {linkage_method}. Metric: {distance}. Number of clusters: {self.n_clusters}.")
            progress_dialog = QtWidgets.QProgressDialog(
                self.tr("Saving clustering result..."), self.tr("Cancel"),
                0, 100, self)
            progress_dialog.setWindowTitle("QGrain")
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            def callback(progress: float):
                if progress_dialog.wasCanceled():
                    raise StopIteration()
                progress_dialog.setValue(int(progress*100))
                QtCore.QCoreApplication.processEvents()
            save_clustering(self.__dataset, flags, filename, progress_callback=callback, logger=self.logger)
        except StopIteration as e:
            self.logger.info("Saving task was canceled.")
            progress_dialog.close()
        except Exception as e:
            progress_dialog.close()
            self.logger.exception("An unknown exception was raised. Please check the logs for more details.", stack_info=True)
            self.show_error(self.tr("An unknown exception was raised. Please check the logs for more details."))

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("Clustering Analyzer"))
        self.linkage_method_label.setText(self.tr("Linkage Method"))
        self.linkage_method_label.setToolTip(self.tr("The linkage method for calculating the distance between the newly formed cluster and each observation vector."))
        self.distance_label.setText(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The distance metric."))
        self.p_label.setText(self.tr("p"))
        self.p_label.setToolTip(self.tr("Controls the number of leaves at the bottom level of the figure."))
        self.n_clusers_label.setText(self.tr("Number of Clusters"))
        self.n_clusers_label.setToolTip(self.tr("Controls the number of clusters of this clustering algorithm."))
