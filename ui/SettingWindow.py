import sys

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtGui import QIcon, QIntValidator, QValidator
from PySide2.QtWidgets import QGridLayout, QMainWindow, QPushButton, QWidget

from ui.AlgorithmSetting import AlgorithmSetting
from ui.AppSetting import AppSetting
from ui.DataSetting import DataSetting


class SettingWindow(QMainWindow):
    sigSaveSettings = Signal(QSettings)
    sigRestoreSettings = Signal(QSettings)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = QSettings("./settings/qgrain.ini", QSettings.Format.IniFormat)
        self.init_ui()
        # self.setWindowFlags(Qt.Drawer | Qt.WindowStaysOnTopHint)
        self.setWindowFlags(Qt.Drawer)
        self.setWindowTitle(self.tr("Settings"))
        self.sigRestoreSettings.connect(self.data_setting.restore_settings)
        self.sigRestoreSettings.connect(self.algorithm_setting.restore_settings)
        self.sigRestoreSettings.connect(self.app_setting.restore_settings)
        self.sigSaveSettings.connect(self.data_setting.save_settings)
        self.sigSaveSettings.connect(self.algorithm_setting.save_settings)
        self.sigSaveSettings.connect(self.app_setting.save_settings)
        self.restore_button.clicked.connect(self.on_restore_clicked)
        self.save_button.clicked.connect(self.on_save_clicked)

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
        self.restore_button = QPushButton(self.tr("Restore"))
        self.save_button = QPushButton(self.tr("Save"))
        self.main_layout.addWidget(self.restore_button, 3, 0)
        self.main_layout.addWidget(self.save_button, 3, 1)

    def setup_all(self):
        self.sigRestoreSettings.emit(self.settings)
        self.sigSaveSettings.emit(self.settings)

    def on_restore_clicked(self):
        self.sigRestoreSettings.emit(self.settings)

    def on_save_clicked(self):
        self.sigSaveSettings.emit(self.settings)
