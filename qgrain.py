import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from multiprocessing import freeze_support

from PySide2.QtCore import QSettings, QTranslator
from PySide2.QtGui import QIcon, QPixmap
from PySide2.QtWidgets import QApplication, QSplashScreen

from ui import GUILogHandler, MainWindow

QGRAIN_VERSION = "0.2.4"

def create_necessary_folders():
    necessary_folders = ("./logs/",)
    for folder in necessary_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

def get_language():
    settings = QSettings("./settings/qgrain.ini", QSettings.Format.IniFormat)
    settings.beginGroup("app")
    lang = settings.value("language")
    settings.endGroup()
    return lang

def get_theme():
    settings = QSettings("./settings/qgrain.ini", QSettings.Format.IniFormat)
    settings.beginGroup("app")
    theme = settings.value("apperance_theme")
    settings.endGroup()
    return theme

def setup_language(app: QApplication):
    lang = get_language()
    trans = QTranslator(app)
    trans.load("./i18n/"+lang)
    app.installTranslator(trans)

def setup_theme(app: QApplication):
    theme = get_theme()
    template_styles = open("./settings/qss/{0}.qss".format(theme)).read()
    custom_style = open("./settings/custom.qss").read()
    app.setStyleSheet(template_styles+custom_style)

def setup_logging(main_window: MainWindow):
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler = TimedRotatingFileHandler("./logs/qgrain.log", when="D", backupCount=256, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(format_str))
    gui_handler = GUILogHandler(main_window)
    gui_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, format=format_str)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("GUI").addHandler(gui_handler)

def main():
    create_necessary_folders()
    app = QApplication(sys.argv)
    logo = QPixmap("./settings/icons/splash_logo.png")
    splash = QSplashScreen(logo)
    splash.show()
    app.processEvents()
    setup_language(app)
    setup_theme(app)
    main_window = MainWindow()
    main_window.setWindowTitle("QGrain")
    main_window.setWindowIcon(QIcon("./settings/icons/icon.png"))
    setup_logging(main_window)
    main_window.show()
    main_window.setup_all()
    splash.finish(main_window)
    sys.exit(app.exec_())

if __name__ == "__main__":
    freeze_support()
    main()
