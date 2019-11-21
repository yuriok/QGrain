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

from ui import MainWindow, GUILogHandler
import logging

# pg.setConfigOptions(background=pg.mkColor("#ffffff00"), foreground=pg.mkColor("#000000ff"))



if __name__ == "__main__":
    # TODO: fix the problem that when use high dpi scaling, the dock bar will not display the title correctly.
    # May be it's related to QSS
    # QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    main_window = MainWindow()


    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(GUILogHandler(main_window))
    
    main_window.show()

    sys.exit(app.exec_())
