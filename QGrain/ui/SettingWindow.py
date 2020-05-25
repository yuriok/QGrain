__all__ = ["SettingWindow"]

import os

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtGui import QIcon, QIntValidator, QValidator
from PySide2.QtWidgets import QGridLayout, QMainWindow, QPushButton, QWidget, QLabel

from QGrain.ui.AlgorithmSettingWidget import AlgorithmSettingWidget
from QGrain.ui.AppSettingWidget import AppSettingWidget
from QGrain.ui.DataSettingWidget import DataSettingWidget

QGRAIN_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


class SettingWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()
        # self.setWindowFlags(Qt.Drawer | Qt.WindowStaysOnTopHint)
        self.setWindowFlags(Qt.Drawer)
        self.setWindowTitle(self.tr("Settings"))

    def init_ui(self):
        self.setting_filename = os.path.join(QGRAIN_ROOT_PATH, "settings", "QGrain.ini")
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout(self.central_widget)

        self.data_settings_title_label = QLabel(self.tr("Data Settings:"))
        self.data_settings_title_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.data_settings_title_label, 0, 0)
        self.data_settings = DataSettingWidget(self.setting_filename, "data_settings")
        self.main_layout.addWidget(self.data_settings, 1, 0)

        self.algorithm_settings_title_label = QLabel(self.tr("Algorithm Settings:"))
        self.algorithm_settings_title_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.algorithm_settings_title_label, 2, 0)
        self.algorithm_settings = AlgorithmSettingWidget(self.setting_filename, "algorithm_settings")
        self.main_layout.addWidget(self.algorithm_settings, 3, 0)

        self.app_settings_title_label = QLabel(self.tr("App Settings:"))
        self.app_settings_title_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.app_settings_title_label, 4, 0)
        self.app_settings = AppSettingWidget(self.setting_filename, "app_settings")
        self.main_layout.addWidget(self.app_settings, 5, 0)

    def setup_all(self):
        self.data_settings.restore()
        self.algorithm_settings.restore()
        self.app_settings.restore()

    def closeEvent(self, e):
        self.data_settings.save()
        self.algorithm_settings.save()
        self.app_settings.save()
        e.accept()
