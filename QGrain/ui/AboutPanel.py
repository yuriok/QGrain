__all__ = ["AboutPanel"]

import os

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QGridLayout, QDialog, QTextBrowser

about_md = """

# QGrain

QGrain aims to provide an easy-to-use and comprehensive analysis platform for grain-size distributions. QGrain has implemented many functions, however, there still are many useful tools that have not been contained. Hence, we published QGrain as an open-source project, and welcome other researchers to contribute their ideas and codes. Codes are available at this GitHub [repository](https://github.com/yuriok/QGrain/). More information and tutorials are available at the [offical website](https://qgrain.net/).

Please cite:

* Liu, Y., Liu, X., Sun, Y., 2021. QGrain: An open-source and easy-to-use software for the comprehensive analysis of grain size distributions. Sedimentary Geology 423, 105980. https://doi.org/10.1016/j.sedgeo.2021.105980

Feel free to contact the author below.

* Yuming Liu, a PhD student of IEECAS, [liuyuming@ieecas.cn](mailto:liuyuming@ieecas.cn)

"""

class AboutPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.init_ui()

    def init_ui(self):
        self.layout = QGridLayout(self)
        self.setWindowTitle(self.tr("About"))
        self.setMinimumSize(400, 400)
        self.text = QTextBrowser()
        self.layout.addWidget(self.text, 0, 0)
        self.text.setMarkdown(about_md)
        self.text.setOpenExternalLinks(True)


if __name__ == "__main__":
    import sys

    from QGrain.entry import setup_app
    app, splash = setup_app()
    main = AboutPanel()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
