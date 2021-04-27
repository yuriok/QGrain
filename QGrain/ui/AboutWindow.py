__all__ = ["AboutWindow"]

import os

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QGridLayout, QMainWindow, QTextBrowser, QWidget
from QGrain import QGRAIN_ROOT_PATH


class AboutWindow(QMainWindow):
    def __init__(self, parent=None):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, flags=flags)
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.setWindowTitle(self.tr("About"))
        self.setMinimumSize(600, 400)
        self.text = QTextBrowser()
        self.layout.addWidget(self.text, 0, 0)
        with open(os.path.join(QGRAIN_ROOT_PATH, "about.md"), mode="r") as text:
            self.text.setMarkdown(text.read())
        self.text.setOpenExternalLinks(True)


if __name__ == "__main__":
    import sys
    from QGrain.entry import setup_app
    app = setup_app()
    main = AboutWindow()
    main.show()
    sys.exit(app.exec_())
