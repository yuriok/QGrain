from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QApplication, QGridLayout, QMainWindow,
                               QTextBrowser, QTextEdit, QWidget)


class AboutWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.setWindowTitle(self.tr("About"))
        # self.setMinimumSize(600, 600)
        self.text = QTextBrowser()

        self.layout.addWidget(self.text, 0, 0)
        self.setWindowFlags(Qt.Drawer)
        self.text.setHtml(
"""
<font face="Arial">
    <h2>QGrain</h2>
    <p>QGrain is an easy to use software that can unmix the multi-modal grain size distribution to some single modals.</p>
    <p>It's writted by Python. This makes it can benefit from the great open source and scientific computation communities.</p>
    <p>QGrain is still during the rapid development stage, its functionalities and usages may changes many and many times. And of course, there probably are some bugs. We are very sorry for its immaturity.</p>
    <p>We are really hope to receive your feedbacks. Whatever it's bug report, request of new feature, disscusion on algorithms.</p>
    <p>Moreover, we are looking forward that there are some partners to join the development of QGrain.</>
    <p>If you have any idea, you can contact the authors below.</p>
    <h4>Authors:</h4>
    <ul>
        <li>Yuming Liu <i><a href="mailto:\\liuyuming@ieecas.cn">liuyuming@ieecas.cn</a></li></i>
    </ul>
</font>
""")
        self.text.setOpenExternalLinks(True)

    def closeEvent(self, e):
        e.ignore()
        self.hide()
        self.saveGeometry()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    s = AboutWindow()
    s.show()
    app.exec_()
