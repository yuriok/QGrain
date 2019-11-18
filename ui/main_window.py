import os
import sys
import time

import numpy as np
import pyqtgraph as pg
import xlrd
from PyQt5.QtCore import QMutex, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
                             QPushButton, QSizePolicy, QSplitter, QWidget)
from pyqtgraph import TableWidget
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter, Parameter

from data import DataLoadParameter
from resolvers import Resolver
from ui import ControlPanel, FittingCanvas


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_ui()
        
        self.fitting_thread = QThread()
        self.timer = QTimer()
        self.resolver = Resolver(emit_iteration=False, display_details=True)


        self.control_panel.sigNcompChanged.connect(self.resolver.on_ncomp_changed)
        self.control_panel.sigNcompChanged.connect(self.canvas.on_ncomp_changed)
        self.control_panel.sigTargetDataChanged.connect(self.resolver.on_target_data_changed)
        self.control_panel.sigTargetDataChanged.connect(self.canvas.on_target_data_changed)
        
        self.resolver.sigEpochFinished.connect(self.canvas.on_epoch_finished)
        self.resolver.sigEpochFinished.connect(self.control_panel.on_epoch_finished)
        self.resolver.sigSingleIterationFinished.connect(self.canvas.on_single_iteration_finished)
        self.data_load_param.sigDataLoaded.connect(self.control_panel.on_data_loaded)
        self.data_load_param.sigDataLoaded.connect(self.on_data_loaded)
        
        self.auto_run_action.triggered.connect(self.control_panel.auto_run)
        self.single_run_action.triggered.connect(self.control_panel.single_run)

        self.canvas_action.triggered.connect(self.show_canvas_dock)
        self.data_loader_action.triggered.connect(self.show_data_loader_dock)
        self.data_preview_action.triggered.connect(self.show_data_preview_dock)

        
        self.resolver.moveToThread(self.fitting_thread)

        self.fitting_thread.start()

        

    
    def init_ui(self):
        
        self.dock_area = DockArea()
        self.setCentralWidget(self.dock_area)

        # Menu
        self.run_menu = self.menuBar().addMenu("Run")
        self.auto_run_action = self.run_menu.addAction("Auto Run")
        self.single_run_action = self.run_menu.addAction("Single Run")
        self.previous_action = self.run_menu.addAction("Previous")
        self.next_action = self.run_menu.addAction("Next")
        self.docks_menu = self.menuBar().addMenu("Docks")
        self.canvas_action = self.docks_menu.addAction("Canvas")
        self.data_loader_action = self.docks_menu.addAction("Load Data")
        self.data_preview_action = self.docks_menu.addAction("Preview Data")
        self.settings_menu = self.menuBar().addMenu("Settings")
        
        # Canvas
        self.canvas_dock = Dock("Canvas", size=(300, 300), closable=True)
        self.dock_area.addDock(self.canvas_dock)
        self.canvas = FittingCanvas()
        self.canvas_dock.addWidget(self.canvas)

        # Control Panel
        self.control_panel_dock = Dock("Control Panel", size=(300, 300), closable=False)
        self.dock_area.addDock(self.control_panel_dock)
        self.control_panel = ControlPanel()
        self.control_panel_dock.addWidget(self.control_panel)

        # Data Loading
        self.data_loader_dock = Dock("Load Data", size=(200, 200), closable=True)
        self.dock_area.addDock(self.data_loader_dock, "right", self.canvas_dock)
        self.data_loader_param_tree = ParameterTree()
        self.data_load_param = DataLoadParameter(name="Parameters to Load Data")
        self.data_loader_param_tree.setParameters(self.data_load_param)
        self.data_loader_dock.addWidget(self.data_loader_param_tree)
        
        # Data Preview
        self.data_preview_dock = Dock("Preview Data", size=(200, 200), closable=True)
        self.dock_area.addDock(self.data_preview_dock, "bottom", self.data_loader_dock)
        self.data_table = TableWidget(editable=False, sortable=False)
        self.data_preview_dock.addWidget(self.data_table)

        self.init_dock_layout()


    def init_dock_layout(self):
        self.dock_area.moveDock(self.canvas_dock, "left", None)
        self.dock_area.moveDock(self.data_preview_dock, "right", self.canvas_dock)
        self.dock_area.moveDock(self.data_loader_dock, "above", self.data_preview_dock)
        self.dock_area.moveDock(self.control_panel_dock, "bottom", self.data_loader_dock)
        


    def show_canvas_dock(self):
        self.dock_area.moveDock(self.canvas_dock, "bottom", None)
        self.canvas_dock.setVisible(True)

    def show_data_loader_dock(self):
        self.dock_area.moveDock(self.data_loader_dock, "bottom", None)
        self.data_loader_dock.setVisible(True)

    def show_data_preview_dock(self):
        self.dock_area.moveDock(self.data_preview_dock, "bottom", None)
        self.data_preview_dock.setVisible(True)


    def on_data_loaded(self, classes, data):
        data_view = []
        for single in data:
            data_view.append([single["id"]] + list(single["data"]))

        self.data_table.setData([["Sample ID"]+list(classes)] + data_view)
