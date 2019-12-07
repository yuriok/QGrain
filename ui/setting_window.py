from PySide2.QtWidgets import QMainWindow, QCheckBox, QLabel, QRadioButton, QPushButton, QGridLayout, QApplication, QSizePolicy, QWidget, QTabWidget, QComboBox, QLineEdit
from PySide2.QtCore import Qt, QSettings, Signal
from PySide2.QtGui import QIcon, QValidator, QIntValidator
import sys

from ui import DataSetting


class SettingWindow(QMainWindow):
    sigSaveSettings = Signal(QSettings)
    sigRestoreSettings = Signal(QSettings)
    def __init__(self):
        super().__init__()
        self.settings = QSettings("./settings/qgrain.ini", QSettings.Format.IniFormat)
        self.init_ui()
        self.setWindowFlags(Qt.Drawer|Qt.WindowStaysOnTopHint)
        self.setWindowTitle(self.tr("Settings"))
        self.sigRestoreSettings.connect(self.data.restore_settings)
        self.sigSaveSettings.connect(self.data.save_settings)
        self.restore_button.clicked.connect(self.on_restore_clicked)
        self.save_button.clicked.connect(self.on_save_clicked)

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout(self.central_widget)
        self.data = DataSetting(self.settings)
        self.data.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        
        self.main_layout.addWidget(self.data, 0, 0, 1, 2)
        self.restore_button = QPushButton(self.tr("Restore"))
        self.save_button = QPushButton(self.tr("Save"))
        self.main_layout.addWidget(self.restore_button, 1, 0)
        self.main_layout.addWidget(self.save_button, 1, 1)


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