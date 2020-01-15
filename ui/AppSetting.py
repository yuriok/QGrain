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
        self.theme_options = [("Aqua", "Aqua"),
                              ("Elegant Dark", "ElegantDark"),
                              ("Material Dark", "MaterialDark"),
                              ("Ubuntu", "Ubuntu")]
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.title_label = QLabel(self.tr("App Settings:"))
        self.title_label.setStyleSheet("QLabel {font: bold;}")
        self.main_layout.addWidget(self.title_label, 0, 0)
        self.language_label = QLabel(self.tr("Language"))
        self.language_combo_box = QComboBox()
        self.language_combo_box.addItems([name for name, enum_value in self.language_options])
        self.language_combo_box.setMaxVisibleItems(5)
        self.theme_label = QLabel(self.tr("Theme"))
        self.theme_combo_box = QComboBox()
        self.theme_combo_box.addItems([name for name, enum_value in self.theme_options])
        self.main_layout.addWidget(self.language_label, 1, 0)
        self.main_layout.addWidget(self.language_combo_box, 1, 1)
        self.main_layout.addWidget(self.theme_label, 2, 0)
        self.main_layout.addWidget(self.theme_combo_box, 2, 1)

    def save_settings(self, settings: QSettings):
        settings.beginGroup("app")
        _, lang = self.language_options[self.language_combo_box.currentIndex()]
        settings.setValue("language", lang)
        self.logger.info("Language has been changed to [%s].", lang)
        _, theme = self.theme_options[self.theme_combo_box.currentIndex()]
        settings.setValue("theme", theme)
        self.logger.info("Theme has been changed to [%s].", theme)
        settings.endGroup()

    def restore_settings(self, settings: QSettings):
        settings.beginGroup("app")
        language = settings.value("language")
        for i, (_, language_value) in enumerate(self.language_options):
            if language == language_value:
                self.language_combo_box.setCurrentIndex(i)
                break
        theme = settings.value("theme")
        for i, (_, theme_value) in enumerate(self.theme_options):
            if theme == theme_value:
                self.theme_combo_box.setCurrentIndex(i)
                break
        settings.endGroup()
