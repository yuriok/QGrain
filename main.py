import logging
import sys
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

from PySide2.QtWidgets import QApplication

from ui import MainWindow, GUILogHandler


if __name__ == "__main__":
    # TODO: fix the problem that when use high dpi scaling, the dock bar will not display the title correctly.
    # May be it's related to QSS
    # QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    print(app)
    main_window = MainWindow()
    # logging
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler = TimedRotatingFileHandler("./logs/qgrain.log", when="D", backupCount=256)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(format_str))
    gui_handler = GUILogHandler(main_window)
    gui_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, format=format_str, handlers=[file_handler])

    main_window.show()
    print(main_window)
    sys.exit(app.exec_())
