__all__ = ["AboutWindow"]

import os

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QGridLayout, QMainWindow, QTextBrowser, QWidget

about_md = """
# QGrain

QGrain is an easy to use software that can unmix the multi-modal grain size distribution to some single modals.

It's written by Python. This makes it can benefit from the great open source and scientific computation communities.

QGrain is still during the rapid development stage, its functionalities and usages may changes many and many times. And of course, there probably are some bugs. We are very sorry for its immaturity.

We are really hope to receive your feedbacks. Whatever it's bug report, request of new feature, disscusion on algorithms.

Moreover, we are looking forward that there are some partners to join the development of QGrain.

If you have any idea, you can contact the authors below.

## Authors

* Yuming Liu

  <a href="mailto:\\liuyuming@ieecas.cn">liuyuming@ieecas.cn</a>

"""

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
        self.text.setMarkdown(about_md)
        self.text.setOpenExternalLinks(True)


if __name__ == "__main__":
    import sys

    from QGrain.entry import setup_app
    app = setup_app()
    main = AboutWindow()
    main.show()
    sys.exit(app.exec_())
