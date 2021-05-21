__all__ = ["AppSettingPanel"]

import os

from PySide2.QtCore import Qt, QSettings
from PySide2.QtWidgets import QGridLayout, QDialog, QCheckBox, QComboBox, QLabel, QPushButton


APP_SETTING_PATH = os.path.join(os.path.expanduser("~"), "QGrain", "app.ini")

class AppSettingPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("App Settings"))
        self.settings = QSettings(APP_SETTING_PATH, QSettings.IniFormat)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)


if __name__ == "__main__":
    import sys

    from QGrain.entry import setup_app
    app, splash = setup_app()
    main = AppSettingPanel()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
