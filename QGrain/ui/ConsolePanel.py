__all__ = ["ConsolePanel"]


import qtawesome as qta
from PySide2.QtCore import QSize, Qt
from PySide2.QtWidgets import QDialog, QGridLayout, QMessageBox, QSizePolicy, QToolButton, QToolButton
from QGrain.ui.AboutWindow import AboutWindow
from QGrain.ui.EMMAResolverPanel import EMMAResolverPanel
from QGrain.ui.GrainSizeDatasetViewer import GrainSizeDatasetViewer
from QGrain.ui.PCAResolverPanel import PCAResolverPanel
from QGrain.ui.RandomDatasetGenerator import RandomDatasetGenerator
from QGrain.ui.SSUAlgorithmTesterPanel import SSUAlgorithmTesterPanel
from QGrain.ui.SSUResolverPanel import SSUResolverPanel
from QGrain.ui.HCResolverPanel import HCResolverPanel
from PySide2.QtGui import QCloseEvent

class ConsolePanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("Console"))
        self.init_ui()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowTitle(self.tr("Warning"))
        self.msg_box.setText(self.tr("Close this window will terminate the work of other windows, are you sure to close it?"))
        self.msg_box.setStandardButtons(QMessageBox.Close | QMessageBox.Cancel)
        self.msg_box.setDefaultButton(QMessageBox.Cancel)
        self.msg_box.setWindowFlags(Qt.Drawer)

    def init_ui(self):
        self.dataset_generator = RandomDatasetGenerator(parent=self)
        self.dataset_viewer = GrainSizeDatasetViewer(parent=self)
        self.pca_resolver = PCAResolverPanel(parent=self)
        self.hc_resolver = HCResolverPanel(parent=self)
        self.emma_resolver = EMMAResolverPanel(parent=self)
        self.ssu_resolver = SSUResolverPanel(parent=self)
        self.ssu_tester = SSUAlgorithmTesterPanel(parent=self)
        self.abount_window = AboutWindow(parent=self)

        self.main_layout = QGridLayout(self)
        self.main_layout.setRowMinimumHeight(0, 120)
        self.main_layout.setRowMinimumHeight(1, 120)
        self.main_layout.setRowMinimumHeight(2, 120)
        self.main_layout.setColumnMinimumWidth(0, 160)
        self.main_layout.setColumnMinimumWidth(1, 160)
        self.main_layout.setColumnMinimumWidth(2, 160)

        self.dataset_generator_button = QToolButton()
        self.dataset_generator_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.dataset_generator_button.setIcon(qta.icon("fa.random"))
        self.dataset_generator_button.setIconSize(QSize(64, 64))
        self.dataset_generator_button.setText(self.tr("Dataset Generator"))
        self.dataset_generator_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.dataset_generator_button.clicked.connect(lambda: self.dataset_generator.show())
        self.dataset_viewer_button = QToolButton()
        self.dataset_viewer_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.dataset_viewer_button.setIcon(qta.icon("fa.table"))
        self.dataset_viewer_button.setIconSize(QSize(64, 64))
        self.dataset_viewer_button.setText(self.tr("Dataset Viewer"))
        self.dataset_viewer_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.dataset_viewer_button.clicked.connect(lambda: self.dataset_viewer.show())
        self.pca_resolver_button = QToolButton()
        self.pca_resolver_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.pca_resolver_button.setIcon(qta.icon("fa.exchange"))
        self.pca_resolver_button.setIconSize(QSize(64, 64))
        self.pca_resolver_button.setText(self.tr("PCA Resolver"))
        self.pca_resolver_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.pca_resolver_button.clicked.connect(lambda: self.pca_resolver.show())
        self.hc_resolver_button = QToolButton()
        self.hc_resolver_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.hc_resolver_button.setIcon(qta.icon("fa.sitemap"))
        self.hc_resolver_button.setIconSize(QSize(64, 64))
        self.hc_resolver_button.setText(self.tr("HC Resolver"))
        self.hc_resolver_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.hc_resolver_button.clicked.connect(lambda: self.hc_resolver.show())
        self.emma_resolver_button = QToolButton()
        self.emma_resolver_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.emma_resolver_button.setIcon(qta.icon("fa.cubes"))
        self.emma_resolver_button.setIconSize(QSize(64, 64))
        self.emma_resolver_button.setText(self.tr("EMMA Resolver"))
        self.emma_resolver_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.emma_resolver_button.clicked.connect(lambda: self.emma_resolver.show())
        self.ssu_resolver_button = QToolButton()
        self.ssu_resolver_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.ssu_resolver_button.setIcon(qta.icon("fa.puzzle-piece"))
        self.ssu_resolver_button.setIconSize(QSize(64, 64))
        self.ssu_resolver_button.setText(self.tr("SSU Resolver"))
        self.ssu_resolver_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.ssu_resolver_button.clicked.connect(lambda: self.ssu_resolver.show())
        self.ssu_tester_button = QToolButton()
        self.ssu_tester_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.ssu_tester_button.setIcon(qta.icon("fa.flask"))
        self.ssu_tester_button.setIconSize(QSize(64, 64))
        self.ssu_tester_button.setText(self.tr("SSU Tester"))
        self.ssu_tester_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.ssu_tester_button.clicked.connect(lambda: self.ssu_tester.show())
        self.setting_button = QToolButton()
        self.setting_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setting_button.setIcon(qta.icon("fa.gears"))
        self.setting_button.setIconSize(QSize(64, 64))
        self.setting_button.setText(self.tr("Setting"))
        self.setting_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.about_button = QToolButton()
        self.about_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.about_button.setIcon(qta.icon("fa.info-circle"))
        self.about_button.setIconSize(QSize(64, 64))
        self.about_button.setText(self.tr("About"))
        self.about_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.about_button.clicked.connect(lambda: self.abount_window.show())

        self.main_layout.addWidget(self.dataset_generator_button, 0, 0)
        self.main_layout.addWidget(self.dataset_viewer_button, 0, 1)
        self.main_layout.addWidget(self.pca_resolver_button, 0, 2)
        self.main_layout.addWidget(self.hc_resolver_button, 1, 0)
        self.main_layout.addWidget(self.emma_resolver_button, 1, 1)
        self.main_layout.addWidget(self.ssu_resolver_button, 1, 2)
        self.main_layout.addWidget(self.ssu_tester_button, 2, 0)
        self.main_layout.addWidget(self.setting_button, 2, 1)
        self.main_layout.addWidget(self.about_button, 2, 2)

    def closeEvent(self, event: QCloseEvent):
        res = self.msg_box.exec_()
        if res == QMessageBox.Close:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    import sys
    from QGrain.entry import setup_app

    app, splash = setup_app()
    main = ConsolePanel()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
