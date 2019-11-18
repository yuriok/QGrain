import sys

import numpy as np
import pyqtgraph as pg
import xlrd
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer,QCoreApplication
from PyQt5.QtWidgets import (QApplication, QGridLayout, QMainWindow,
                             QSizePolicy, QSplitter, QWidget)
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter, Parameter
from scipy.optimize import minimize

from ui import MainWindow

# pg.setConfigOptions(background=pg.mkColor("#ffffff00"), foreground=pg.mkColor("#000000ff"))



if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.setGeometry(200, 200, 900, 600)
    mainWindow.show()

    sys.exit(app.exec_())
