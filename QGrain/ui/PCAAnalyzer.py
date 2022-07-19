__all__ = ["PCAAnalyzer"]

from PySide6 import QtCore, QtWidgets

from ..models import Dataset
from ..charts.PCAResultChart import PCAResultChart


class PCAAnalyzer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("PCA Analyzer"))
        self.main_layout = QtWidgets.QGridLayout(self)
        self.chart = PCAResultChart()
        self.main_layout.addWidget(self.chart, 0, 0)
        self.file_dialog = QtWidgets.QFileDialog(self)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self._dataset = None

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

    def on_dataset_loaded(self, dataset: Dataset):
        self._dataset = dataset
        self.chart.show_dataset(self._dataset)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("PCA Analyzer"))
