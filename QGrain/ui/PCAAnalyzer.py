__all__ = ["PCAAnalyzer"]

import typing

from PySide6 import QtCore, QtWidgets

from ..chart.PCAResultChart import PCAResultChart
from ..model import GrainSizeDataset


class PCAAnalyzer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("PCA Resolver"))
        self.init_ui()
        self.file_dialog = QtWidgets.QFileDialog(self)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.__dataset = None

    def init_ui(self):
        self.main_layout = QtWidgets.QGridLayout(self)
        self.chart = PCAResultChart()
        self.main_layout.addWidget(self.chart, 0, 0)

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

    def on_dataset_loaded(self, dataset: GrainSizeDataset):
        self.__dataset = dataset
        self.chart.show_dataset(dataset)
