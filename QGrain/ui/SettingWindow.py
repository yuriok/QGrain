__all__ = ["SettingWindow"]

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtGui import QIcon, QIntValidator, QValidator
from PySide2.QtWidgets import QGridLayout, QMainWindow, QPushButton, QWidget

from QGrain.ui.AlgorithmSetting import AlgorithmSetting
from QGrain.ui.AppSetting import AppSetting
from QGrain.ui.DataSetting import DataSetting


class SettingWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()
        # self.setWindowFlags(Qt.Drawer | Qt.WindowStaysOnTopHint)
        self.setWindowFlags(Qt.Drawer)
        self.setWindowTitle(self.tr("Settings"))

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout(self.central_widget)
        self.data_setting = DataSetting()
        self.main_layout.addWidget(self.data_setting, 0, 0, 1, 2)
        self.algorithm_setting = AlgorithmSetting()
        self.main_layout.addWidget(self.algorithm_setting, 1, 0, 1, 2)
        self.app_setting = AppSetting()
        self.main_layout.addWidget(self.app_setting, 2, 0, 1, 2)

    def setup_all(self):
        self.data_setting.restore()
        self.algorithm_setting.restore()
        self.app_setting.restore()
