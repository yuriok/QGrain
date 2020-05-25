__all__ = ["AppSettingWidget"]

import logging


from PySide2.QtCore import QSettings, Qt, Signal
from PySide2.QtWidgets import QComboBox, QGridLayout, QLabel, QWidget


class AppSettingWidget(QWidget):
    logger = logging.getLogger("root.ui.AppSettingWidget")
    gui_logger = logging.getLogger("GUI")
    def __init__(self, filename: str = None, group: str = None):
        super().__init__()
        self.language_options = [("简体中文", "zh_CN"),
                                 ("English", "en")]
        self.theme_options = [("Aqua", "Aqua"),
                              ("Elegant Dark", "ElegantDark"),
                              ("Material Dark", "MaterialDark"),
                              ("Ubuntu", "Ubuntu")]
        if filename is not None:
            self.setting_file = QSettings(filename, QSettings.Format.IniFormat)
            if group is not None:
                self.setting_file.beginGroup(group)
        else:
            self.setting_file = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.initialize_ui()

    def initialize_ui(self):
        self.main_layout = QGridLayout(self)
        self.language_label = QLabel(self.tr("Language"))
        self.language_combo_box = QComboBox()
        self.language_combo_box.addItems([name for name, enum_value in self.language_options])
        self.language_combo_box.setMaxVisibleItems(5)
        self.theme_label = QLabel(self.tr("Theme"))
        self.theme_combo_box = QComboBox()
        self.theme_combo_box.addItems([name for name, enum_value in self.theme_options])
        self.main_layout.addWidget(self.language_label, 0, 0)
        self.main_layout.addWidget(self.language_combo_box, 0, 1)
        self.main_layout.addWidget(self.theme_label, 1, 0)
        self.main_layout.addWidget(self.theme_combo_box, 1, 1)

        self.language_combo_box.currentIndexChanged.connect(self.on_language_changed)
        self.theme_combo_box.currentIndexChanged.connect(self.on_theme_changed)

    def on_language_changed(self, index: int):
        _, lang = self.language_options[index]
        self.logger.info("Language has been changed to [%s].", lang)

    def on_theme_changed(self, index: int):
        _, theme = self.theme_options[index]
        self.logger.info("Theme has been changed to [%s].", theme)

    def save(self):
        if self.setting_file is not None:
            _, lang = self.language_options[self.language_combo_box.currentIndex()]
            self.setting_file.setValue("language", lang)
            _, theme = self.theme_options[self.theme_combo_box.currentIndex()]
            self.setting_file.setValue("theme", theme)
            
            self.logger.info("App settings have been saved to the file.")

    def restore(self):
        if self.setting_file is not None:
            language = self.setting_file.value("language", defaultValue="en", type=str)
            for i, (_, language_value) in enumerate(self.language_options):
                if language == language_value:
                    self.language_combo_box.setCurrentIndex(i)
                    break
            theme = self.setting_file.value("theme", defaultValue="MaterialDark", type=str)
            for i, (_, theme_value) in enumerate(self.theme_options):
                if theme == theme_value:
                    self.theme_combo_box.setCurrentIndex(i)
                    break

            self.logger.info("App settings have been retored from the file.")
