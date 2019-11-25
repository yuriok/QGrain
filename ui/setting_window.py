from PySide2.QtWidgets import QMainWindow, QCheckBox, QLabel, QRadioButton, QPushButton, QGridLayout, QApplication, QSizePolicy, QWidget, QTabWidget



class SettingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.tab = QTabWidget(self)
        self.setCentralWidget(self.tab)
        self.tab.setTabPosition(QTabWidget.TabPosition.West)
        
        self.data = DataSetting()
        self.tab.addTab(self.data, "add")


class DataSetting(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(QPushButton("Hello1"), 0, 0)
        self.main_layout.addWidget(QPushButton("Hello2"), 0, 1)
        self.main_layout.addWidget(QPushButton("Hello3"), 0, 2)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    s = SettingWindow()
    s.show()
    app.exec_()