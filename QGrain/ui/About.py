__all__ = ["About"]

import os

from PySide6 import QtCore, QtWidgets

from . import QGRAIN_ROOT_PATH


class About(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("About"))
        self.setMinimumSize(400, 400)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.text = QtWidgets.QTextBrowser()
        self.text.setReadOnly(True)
        self.text.setOpenExternalLinks(True)
        self.main_layout.addWidget(self.text, 0, 0)
        with open(os.path.join(QGRAIN_ROOT_PATH, "assets", "README.md"), "r") as f:
            md = f.read()
            self.text.setMarkdown(md)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("About"))
