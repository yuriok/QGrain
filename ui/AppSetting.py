import logging

from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtWidgets import QComboBox, QGridLayout, QLabel, QWidget


class AppSetting(QWidget):
    logger = logging.getLogger("root.ui.AppSetting")
    gui_logger = logging.getLogger("GUI")
    def __init__(self):
        super().__init__()
        self.language_options = [("简体中文", "zh_CN"),
                                 ("English", "en")]
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.title_label = QLabel(self.tr("App Settings:"))
        self.title_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.title_label, 0, 0)
        self.language_label = QLabel(self.tr("Language"))
        self.language_combox = QComboBox()
        self.language_combox.addItems([name for name, enum_value in self.language_options])
        self.language_combox.setMaxVisibleItems(5)
        self.main_layout.addWidget(self.language_label, 1, 0)
        self.main_layout.addWidget(self.language_combox, 1, 1)

    def save_settings(self, settings: QSettings):
        settings.beginGroup("app")
        name, lang = self.language_options[self.language_combox.currentIndex()]
        settings.setValue("language", lang)
        self.logger.info("Language has been changed to [%s].", lang)
        settings.endGroup()

    def restore_settings(self, settings: QSettings):
        settings.beginGroup("app")
        language = settings.value("language")
        for i, (name, lang) in enumerate(self.language_options):
            if language == lang:
                self.language_combox.setCurrentIndex(i)
                break
        settings.endGroup()
