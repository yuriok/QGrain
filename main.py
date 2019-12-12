import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from PySide2.QtGui import QFont, QIcon
from PySide2.QtCore import QCoreApplication, Qt, QTextCodec
from PySide2.QtWidgets import QApplication

from ui import GUILogHandler, MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setWindowTitle("QGrain")
    main_window.setWindowIcon(QIcon("./settings/icons/icon.png"))
    # use qss
    template_styles = open("./settings/qss/Ubuntu.qss").read()
    custom_style = open("./settings/custom.qss").read()
    app.setStyleSheet(template_styles+custom_style)
    # logging
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler = TimedRotatingFileHandler("./logs/qgrain.log", when="D", backupCount=256)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(format_str))
    gui_handler = GUILogHandler(main_window)
    gui_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, format=format_str)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("GUI").addHandler(gui_handler)
    main_window.show()
    # TODO: use interface
    main_window.control_panel.init_conditions()
    main_window.settings_window.init_settings()
    sys.exit(app.exec_())


#  translator = QtCore.QTranslator()
#  translator.load('i18n/eo_EO')
#  app = QtGui.QApplication(sys.argv)
#  app.installTranslator(translator)