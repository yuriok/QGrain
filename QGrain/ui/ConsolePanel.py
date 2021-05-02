__all__ = ["ConsolePanel"]


import qtawesome as qta
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout, QMessageBox, QPushButton
from QGrain.ui.AboutWindow import AboutWindow
from QGrain.ui.EMMAResolverPanel import EMMAResolverPanel
from QGrain.ui.GrainSizeDatasetViewer import GrainSizeDatasetViewer
from QGrain.ui.PCAResolverPanel import PCAResolverPanel
from QGrain.ui.RandomDatasetGenerator import RandomDatasetGenerator
from QGrain.ui.SSUAlgorithmTesterPanel import SSUAlgorithmTesterPanel
from QGrain.ui.SSUResolverPanel import SSUResolverPanel
from QGrain.ui.HierarchyResolverPanel import HierarchyResolverPanel
from PySide2.QtGui import QCloseEvent

class ConsolePanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("Console"))
        self.init_ui()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowTitle(self.tr("Warning"))
        self.msg_box.setText("Close this window will terminate the work of other windows, are you sure to close it?")
        self.msg_box.setStandardButtons(QMessageBox.Close | QMessageBox.Cancel)
        self.msg_box.setDefaultButton(QMessageBox.Cancel)
        self.msg_box.setWindowFlags(Qt.Drawer)

    def init_ui(self):
        self.random_generator = RandomDatasetGenerator(parent=self)
        self.dataset_viewer = GrainSizeDatasetViewer(parent=self)
        self.pca_resolver = PCAResolverPanel(parent=self)
        self.hierarchy_resolver = HierarchyResolverPanel(parent=self)
        self.emma_resolver = EMMAResolverPanel(parent=self)
        self.ssu_resolver = SSUResolverPanel(parent=self)
        self.ssu_algorithm_tester = SSUAlgorithmTesterPanel(parent=self)
        self.abount_window = AboutWindow(parent=self)

        self.main_layout = QGridLayout(self)
        self.random_generator_button = QPushButton(qta.icon("fa.random"), self.tr("Random Dataset Generator"))
        self.random_generator_button.clicked.connect(lambda: self.random_generator.show())
        self.dataset_viewer_button = QPushButton(qta.icon("fa.table"), self.tr("Dataset Viewer"))
        self.dataset_viewer_button.clicked.connect(lambda: self.dataset_viewer.show())
        self.pca_resolver_button = QPushButton(qta.icon("fa.exchange"), self.tr("PCA Resolver"))
        self.pca_resolver_button.clicked.connect(lambda: self.pca_resolver.show())
        self.hierarchy_resolver_button = QPushButton(qta.icon("fa.sitemap"), self.tr("Hierarchy Clustering Resolver"))
        self.hierarchy_resolver_button.clicked.connect(lambda: self.hierarchy_resolver.show())
        self.emma_resolver_button = QPushButton(qta.icon("fa.cubes"), self.tr("EMMA Resolver"))
        self.emma_resolver_button.clicked.connect(lambda: self.emma_resolver.show())
        self.ssu_resolver_button = QPushButton(qta.icon("fa.puzzle-piece"), self.tr("SSU Resolver"))
        self.ssu_resolver_button.clicked.connect(lambda: self.ssu_resolver.show())
        self.ssu_algorithm_tester_button = QPushButton(qta.icon("fa.flask"), self.tr("SSU Algorithm Tester"))
        self.ssu_algorithm_tester_button.clicked.connect(lambda: self.ssu_algorithm_tester.show())
        # self.setting_button = QPushButton(qta.icon("fa.gears"), self.tr("Setting"))
        self.about_button = QPushButton(qta.icon("fa.info-circle"), self.tr("About"))
        self.about_button.clicked.connect(lambda: self.abount_window.show())

        self.main_layout.addWidget(self.random_generator_button, 0, 0)
        self.main_layout.addWidget(self.dataset_viewer_button, 1, 0)
        self.main_layout.addWidget(self.pca_resolver_button, 2, 0)
        self.main_layout.addWidget(self.hierarchy_resolver_button, 3, 0)
        self.main_layout.addWidget(self.emma_resolver_button, 4, 0)
        self.main_layout.addWidget(self.ssu_resolver_button, 5, 0)
        self.main_layout.addWidget(self.ssu_algorithm_tester_button, 6, 0)
        # self.main_layout.addWidget(self.setting_button, 7, 0)
        self.main_layout.addWidget(self.about_button, 8, 0)

    def closeEvent(self, event: QCloseEvent):
        res = self.msg_box.exec_()
        if res == QMessageBox.Close:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    import sys
    from QGrain.entry import setup_app

    app = setup_app()
    main = ConsolePanel()
    main.show()
    sys.exit(app.exec_())
