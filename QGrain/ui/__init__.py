import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

import matplotlib as mpl
from PySide6 import QtWidgets, QtCore, QtGui

from .RuntimeLog import StatusBarLogHandler, GUILogHandler, RuntimeLog
from .. import QGRAIN_VERSION, QGRAIN_ROOT_PATH
from ..charts import setup_matplotlib

EXTRA = {"font_family": "Source Han Sans CN",
         "density_scale": "-1"}

def get_dir_size(directory: str):
    size = 0
    for root, _, files in os.walk(directory):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size


def create_necessary_folders():
    necessary_folders = (
        os.path.join(os.path.expanduser("~"), "QGrain"),
        os.path.join(os.path.expanduser("~"), "QGrain", "logs"))
    for folder in necessary_folders:
        os.makedirs(folder, exist_ok=True)


def setup_language(app: QtWidgets.QApplication, language: str):
    trans = QtCore.QTranslator(app)
    trans.load(os.path.join(QGRAIN_ROOT_PATH, "assets", language))
    app.installTranslator(trans)


def setup_logging(status_bar: QtWidgets.QStatusBar, log_dialog: RuntimeLog):
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler = TimedRotatingFileHandler(os.path.join(os.path.expanduser("~"), "QGrain", "logs", "qgrain.log"),
                                            when="D", backupCount=8, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(format_str))
    status_bar_handler = StatusBarLogHandler(status_bar, level=logging.INFO)
    status_bar_handler.setFormatter(logging.Formatter(format_str))
    gui_handler = GUILogHandler(log_dialog, level=logging.DEBUG)
    gui_handler.setFormatter(logging.Formatter(format_str))
    logging.basicConfig(level=logging.DEBUG, format=format_str)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(status_bar_handler)
    logging.getLogger().addHandler(gui_handler)
    mpl.set_loglevel("error")


def setup_app(language="en", theme="default"):
    assert language in ("en", "zh_CN")
    assert theme in ("default", "light", "dark")

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    fonts_path = os.path.join(QGRAIN_ROOT_PATH, "assets", "fonts")
    for font in os.listdir(fonts_path):
        QtGui.QFontDatabase.addApplicationFont(os.path.join(fonts_path, font))
    pixmap = QtGui.QPixmap(os.path.join(QGRAIN_ROOT_PATH, "assets", "icon.png"))
    create_necessary_folders()
    app.setWindowIcon(QtGui.QIcon(pixmap))
    app.setApplicationVersion(QGRAIN_VERSION)
    from qt_material import apply_stylesheet
    apply_stylesheet(app, theme=os.path.join(QGRAIN_ROOT_PATH, "assets", f"{theme}_theme.xml"),
                     invert_secondary=True, extra=EXTRA)
    setup_matplotlib()
    setup_language(app, f"{language}")
    return app
