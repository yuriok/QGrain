from PySide2.QtWidgets import QMainWindow, QCheckBox, QLabel, QRadioButton, QPushButton, QGridLayout, QApplication, QSizePolicy, QWidget, QTabWidget, QComboBox, QLineEdit
from PySide2.QtCore import Qt, QSettings, Signal
from PySide2.QtGui import QIcon, QValidator, QIntValidator
import sys

from ui import AlgorithmSetting, DataSetting, AppSetting
# from data_setting import DataSetting
# from algorithm_setting import AlgorithmSetting
# from app_setting import AppSetting

class SettingWindow(QMainWindow):
    sigSaveSettings = Signal(QSettings)
    sigRestoreSettings = Signal(QSettings)
    def __init__(self):
        super().__init__()
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
        self.data_setting.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.main_layout.addWidget(self.data_setting, 0, 0, 1, 2)
        self.algorithm_setting = AlgorithmSetting()
        self.algorithm_setting.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.main_layout.addWidget(self.algorithm_setting, 1, 0, 1, 2)
        self.app_setting = AppSetting()
        self.app_setting.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.main_layout.addWidget(self.app_setting, 2, 0, 1, 2)
        
        self.restore_button = QPushButton(self.tr("Restore"))
        self.save_button = QPushButton(self.tr("Save"))
        self.main_layout.addWidget(self.restore_button, 3, 0)
        self.main_layout.addWidget(self.save_button, 3, 1)


    def init_settings(self):
        self.sigRestoreSettings.emit(self.settings)
        self.sigSaveSettings.emit(self.settings)

    def on_restore_clicked(self):
        self.sigRestoreSettings.emit(self.settings)

    def on_save_clicked(self):
        self.sigSaveSettings.emit(self.settings)



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    template_styles = open("./settings/qss/aqua.qss").read()
    custom_style = open("./settings/custom.qss").read()
    app.setStyleSheet(template_styles+custom_style)
    s = SettingWindow()
    s.init_settings()
    s.show()
    app.exec_()