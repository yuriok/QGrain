from PySide2.QtWidgets import QMainWindow, QCheckBox, QLabel, QRadioButton, QPushButton, QGridLayout, QApplication, QSizePolicy, QWidget, QTabWidget, QComboBox, QLineEdit, QMessageBox
from PySide2.QtCore import Qt, QSettings, Signal
from PySide2.QtGui import QIcon,QValidator,QIntValidator

import logging

class AlgorithmSetting(QWidget):
    sigResovlerSettingChanged = Signal(dict)
    logger = logging.getLogger("root.ui.AlgorithmSetting")
    def __init__(self, settings: QSettings):
        super().__init__()
        self.settings = settings
        self.msg_box = QMessageBox()
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.init_ui()