from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (QDialog, QGridLayout, QMessageBox, QPushButton,
                               QSizePolicy)

from .AboutDialog import AboutDialog
from .ClusteringAnalyzer import ClusteringAnalyzer
from .DatasetGenerator import DatasetGenerator
from .EMMAAnalyzer import EMMAAnalyzer
from .GrainSizeDatasetViewer import GrainSizeDatasetViewer
from .PCAAnalyzer import PCAAnalyzer
from .ReferenceAssembler import ReferenceAssembler
from .SSUAnalyzer import SSUAnalyzer


class ConsolePanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("Console"))
        self.init_ui()
        self.normal_msg = QMessageBox(self)
        self.normal_msg.setWindowTitle(self.tr("Warning"))
        self.normal_msg.setText(self.tr("Close this window will terminate the work of other windows, are you sure to close it?"))
        self.normal_msg.setStandardButtons(QMessageBox.Close | QMessageBox.Cancel)
        self.normal_msg.setDefaultButton(QMessageBox.Cancel)

    def init_ui(self):
        self.dataset_generator = DatasetGenerator(parent=self)
        self.dataset_viewer = GrainSizeDatasetViewer(parent=self)
        self.pca_resolver = PCAAnalyzer(parent=self)
        self.hc_resolver = ClusteringAnalyzer(parent=self)
        self.emma_resolver = EMMAAnalyzer(parent=self)
        self.ssu_resolver = SSUAnalyzer(parent=self)
        self.abount_window = AboutDialog(parent=self)

        self.main_layout = QGridLayout(self)
        self.main_layout.setRowMinimumHeight(0, 120)
        self.main_layout.setRowMinimumHeight(1, 120)
        self.main_layout.setColumnMinimumWidth(0, 160)
        self.main_layout.setColumnMinimumWidth(1, 160)
        self.main_layout.setColumnMinimumWidth(2, 160)
        self.main_layout.setColumnMinimumWidth(3, 160)

        self.dataset_generator_button = QPushButton(self.tr("Dataset Generator"))
        self.dataset_generator_button.clicked.connect(lambda: self.dataset_generator.show())
        self.dataset_viewer_button = QPushButton(self.tr("Dataset Viewer"))
        self.dataset_viewer_button.clicked.connect(lambda: self.dataset_viewer.show())
        self.pca_resolver_button = QPushButton(self.tr("PCA Resolver"))
        self.pca_resolver_button.clicked.connect(lambda: self.pca_resolver.show())
        self.hc_resolver_button = QPushButton(self.tr("HC Resolver"))
        self.hc_resolver_button.clicked.connect(lambda: self.hc_resolver.show())
        self.emma_resolver_button = QPushButton(self.tr("EMMA Resolver"))
        self.emma_resolver_button.clicked.connect(lambda: self.emma_resolver.show())
        self.ssu_resolver_button = QPushButton(self.tr("SSU Resolver"))
        self.ssu_resolver_button.clicked.connect(lambda: self.ssu_resolver.show())
        self.about_button = QPushButton(self.tr("About"))
        self.about_button.clicked.connect(lambda: self.abount_window.show())

        self.main_layout.addWidget(self.dataset_generator_button, 0, 0)
        self.main_layout.addWidget(self.dataset_viewer_button, 0, 1)
        self.main_layout.addWidget(self.pca_resolver_button, 0, 2)
        self.main_layout.addWidget(self.hc_resolver_button, 0, 3)
        self.main_layout.addWidget(self.emma_resolver_button, 1, 0)
        self.main_layout.addWidget(self.ssu_resolver_button, 1, 1)
        self.main_layout.addWidget(self.about_button, 1, 2)

    def closeEvent(self, event: QCloseEvent):
        res = self.normal_msg.exec_()
        if res == QMessageBox.Close:
            event.accept()
        else:
            event.ignore()
