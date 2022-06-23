import os

from PySide6 import QtCore, QtWidgets, QtGui

about_md = """

# QGrain

QGrain aims to provide an easy-to-use and comprehensive analysis platform for grain size distributions. QGrain has implemented many functions, however, there still are many useful tools that have not been contained. Hence, we published QGrain as an open-source project, and welcome other researchers to contribute their ideas and codes. Codes are available at this GitHub [repository](https://github.com/yuriok/QGrain/). More information and tutorials are available at the [offical website](https://qgrain.net/).

Please cite:

* Liu, Y., Liu, X., Sun, Y., 2021. QGrain: An open-source and easy-to-use software for the comprehensive analysis of grain size distributions. Sedimentary Geology 423, 105980. https://doi.org/10.1016/j.sedgeo.2021.105980

Feel free to contact the author below.

* Yuming Liu, a PhD student of IEECAS, [liuyuming@ieecas.cn](mailto:liuyuming@ieecas.cn)

"""

class AboutDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=QtCore.Qt.Window)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("About"))
        self.setMinimumSize(400, 400)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.text = QtWidgets.QTextBrowser()
        self.text.setReadOnly(True)
        self.text.setMarkdown(self.tr(about_md))
        self.text.setOpenExternalLinks(True)
        self.main_layout.addWidget(self.text, 0, 0)


    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("About"))
