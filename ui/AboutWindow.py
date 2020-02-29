from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QGridLayout, QMainWindow, QTextBrowser,
                               QTextEdit, QWidget)


class AboutWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.setWindowTitle(self.tr("About"))
        self.setMinimumSize(600, 400)
        self.text = QTextBrowser()
        self.layout.addWidget(self.text, 0, 0)
        self.setWindowFlags(Qt.Drawer)
        self.text.setSource("./settings/about.md")
        self.text.setOpenExternalLinks(True)


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    s = AboutWindow()
    s.show()
    app.exec_()
