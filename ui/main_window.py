import os
import sys
import time

import numpy as np
import pyqtgraph as pg
import xlrd

from PyQt5.QtGui import QCursor
from PyQt5.QtCore import QMutex, Qt, QThread, QTimer, pyqtSignal, QModelIndex
from PyQt5.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,QTableWidget,QTableWidgetItem,QAbstractItemView,
                             QPushButton, QSizePolicy, QSplitter, QWidget, QMenu)
from pyqtgraph import TableWidget
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter, Parameter

from data import GrainSizeData, FittedData
from resolvers import Resolver
from ui import ControlPanel, FittingCanvas
from queue import Queue
from data_manager import DataManager
import logging



class GUILogHandler(logging.Handler):
    def __init__(self, target_widget):
        logging.Handler.__init__(self)
        self.target = target_widget
        self.__mutex = QMutex()

    def emit(self, record):
        self.__mutex.lock()
        if record.levelno < self.level:
            return
        message = "{0} - {1} - {2} - {3}".format(record.asctime, record.name, record.levelname, record.msg)
        self.target.show_message(message)
        self.__mutex.unlock()


class MainWindow(QMainWindow):
    sigDataSelected = pyqtSignal(int)
    sigRemoveRecords = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.status_bar = self.statusBar()
        self.recorded_data_count = 0
        self.fitting_thread = QThread()
        self.resolver = Resolver(display_details=True)
        self.resolver.moveToThread(self.fitting_thread)
        self.fitting_thread.start()
        self.data_manager = DataManager()
        self.connect_all()

        
    
    
    def show_message(self, message):
        self.status_bar.showMessage(message)

    
    def init_ui(self):
        
        self.dock_area = DockArea()
        self.setCentralWidget(self.dock_area)

        # Menu
        self.file_menu = self.menuBar().addMenu("File")
        self.load_action = self.file_menu.addAction("Load")
        self.save_action = self.file_menu.addAction("Save")
        self.docks_menu = self.menuBar().addMenu("Docks")
        self.canvas_action = self.docks_menu.addAction("Canvas")
        self.control_panel_action = self.docks_menu.addAction("Control Panel")
        self.raw_data_table_action = self.docks_menu.addAction("Raw Data Table")
        self.recorded_data_table_action = self.docks_menu.addAction("Recorded Data Table")
        self.reset_docks_actions = self.docks_menu.addAction("Reset")
        self.settings_menu = self.menuBar().addMenu("Settings")
        
        # Canvas
        self.canvas_dock = Dock("Canvas", size=(200, 300), closable=True)
        self.dock_area.addDock(self.canvas_dock)
        self.canvas = FittingCanvas()
        self.canvas_dock.addWidget(self.canvas)


        # Control Panel
        self.control_panel_dock = Dock("Control Panel", size=(200, 100), closable=True)
        self.control_panel_dock.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.dock_area.addDock(self.control_panel_dock)
        self.control_panel = ControlPanel()
        self.control_panel_dock.addWidget(self.control_panel)

        
        # Raw Data Table
        self.raw_data_dock = Dock("Raw Data Table", size=(300, 400), closable=True)
        self.dock_area.addDock(self.raw_data_dock)
        self.raw_data_table = QTableWidget()
        self.raw_data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.raw_data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.raw_data_table.setAlternatingRowColors(True)
        self.raw_data_dock.addWidget(self.raw_data_table)

        # Recorded Data Table
        self.recorded_data_dock = Dock("Recorded Data Table", size=(300, 400), closable=True)
        self.dock_area.addDock(self.recorded_data_dock)
        self.recorded_data_table = QTableWidget()
        self.recorded_data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.recorded_data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.recorded_data_table.setAlternatingRowColors(True)
        self.recorded_data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.recorded_data_dock.addWidget(self.recorded_data_table)
        self.recorded_table_menu = QMenu(self.recorded_data_table)
        self.recorded_table_remove_action = self.recorded_table_menu.addAction("Remove")
        
        
        self.reset_dock_layout()


    def connect_all(self):
        # TODO: Type Switch
        self.control_panel.sigDistributionTypeChanged.connect(self.resolver.on_type_changed)
        self.control_panel.sigNcompChanged.connect(self.resolver.on_ncomp_changed)
        self.control_panel.sigNcompChanged.connect(self.canvas.on_ncomp_changed)
        self.control_panel.sigFocusSampleChanged.connect(self.data_manager.on_focus_sample_changed)
        self.control_panel.sigFocusSampleChanged.connect(self.on_focus_sample_changed)
        self.control_panel.sigResolverSettingsChanged.connect(self.resolver.on_settings_changed)
        self.control_panel.sigRuningSettingsChanged.connect(self.on_settings_changed)
        self.control_panel.sigDataSettingsChanged.connect(self.data_manager.on_settings_changed)
        self.control_panel.try_fit_button.clicked.connect(self.resolver.try_fit)
        self.control_panel.record_button.clicked.connect(self.data_manager.record_data)
        
        self.data_manager.sigDataLoaded.connect(self.on_data_loaded)
        self.data_manager.sigDataLoaded.connect(self.control_panel.on_data_loaded)
        self.data_manager.sigTargetDataChanged.connect(self.resolver.on_target_data_changed)
        self.data_manager.sigTargetDataChanged.connect(self.canvas.on_target_data_changed)
        self.data_manager.sigDataRecorded.connect(self.on_data_recorded)
        

        self.resolver.sigEpochFinished.connect(self.control_panel.on_epoch_finished)
        self.resolver.sigEpochFinished.connect(self.canvas.on_epoch_finished)
        self.resolver.sigEpochFinished.connect(self.data_manager.on_epoch_finished)
        self.resolver.sigSingleIterationFinished.connect(self.canvas.on_single_iteration_finished)
        
        
        self.canvas.sigWidgetsEnable.connect(self.control_panel.on_widgets_enable_changed)
        
        self.load_action.triggered.connect(self.data_manager.on_load_data_clicked)
        self.canvas_action.triggered.connect(self.show_canvas_dock)
        self.control_panel_action.triggered.connect(self.show_control_panel_dock)
        self.raw_data_table_action.triggered.connect(self.show_raw_data_dock)
        self.recorded_data_table_action.triggered.connect(self.show_recorded_data_dock)
        self.reset_docks_actions.triggered.connect(self.reset_dock_layout)

        self.raw_data_table.cellClicked.connect(self.on_data_item_clicked)
        self.recorded_table_remove_action.triggered.connect(self.remove_recorded_selection)
        self.recorded_data_table.customContextMenuRequested.connect(self.show_recorded_table_menu)
        self.sigDataSelected.connect(self.control_panel.on_data_selected)
        self.sigRemoveRecords.connect(self.data_manager.remove_data)




    def reset_dock_layout(self):
        self.dock_area.moveDock(self.canvas_dock, "left", None)
        self.dock_area.moveDock(self.raw_data_dock, "right", self.canvas_dock)
        self.dock_area.moveDock(self.recorded_data_dock, "below", self.raw_data_dock)
        self.dock_area.moveDock(self.control_panel_dock, "bottom", self.canvas_dock)


    def show_canvas_dock(self):
        self.dock_area.moveDock(self.canvas_dock, "bottom", None)
        self.canvas_dock.setVisible(True)

    def show_control_panel_dock(self):
        self.dock_area.moveDock(self.control_panel_dock, "bottom", None)
        self.control_panel_dock.setVisible(True)

    def show_raw_data_dock(self):
        self.dock_area.moveDock(self.raw_data_dock, "bottom", None)
        self.raw_data_dock.setVisible(True)

    def show_recorded_data_dock(self):
        self.dock_area.moveDock(self.recorded_data_dock, "bottom", None)
        self.recorded_data_dock.setVisible(True)


    def on_data_loaded(self, grain_size_data: GrainSizeData):
        nrows = len(grain_size_data.sample_data_list)
        ncols = len(grain_size_data.classes)
        self.raw_data_table.setRowCount(nrows)
        self.raw_data_table.setColumnCount(ncols)

        self.raw_data_table.setHorizontalHeaderLabels(["{0:.4f}".format(class_value) for class_value in grain_size_data.classes])
        self.raw_data_table.setVerticalHeaderLabels([str(sample_data.name) for sample_data in grain_size_data.sample_data_list])
        # self.data_table.setData(grain_size_data.table_view)
        for i, sample_data in enumerate(grain_size_data.sample_data_list):
            for j, value in enumerate(sample_data.distribution):
                item = QTableWidgetItem("{0:.4f}".format(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.raw_data_table.setItem(i, j, item)

    def on_data_recorded(self, fitted_data: FittedData):
        # the 2 additional rows are headers
        if self.recorded_data_count + 2 >= self.recorded_data_table.rowCount():
            self.recorded_data_table.setRowCount(self.recorded_data_table.rowCount() + 50)
        ncomp = len(fitted_data.statistic)
        if ncomp*9+2 > self.recorded_data_table.columnCount():
            self.recorded_data_table.setColumnCount(ncomp*9+2)

        def write(row, col, value, e=False):
            if type(value) == str:
                item = QTableWidgetItem(value) 
            else:
                if e:
                    item = QTableWidgetItem("{0:.4e}".format(value))
                else:
                    item = QTableWidgetItem("{0:.4f}".format(value))
            item.setTextAlignment(Qt.AlignCenter)
            self.recorded_data_table.setItem(row, col, item)

        # Write Header
        write(0, 0, "Sample Name")
        self.recorded_data_table.setSpan(0, 0, 2, 1)
        write(0, 1, "Mean Squared Error")
        self.recorded_data_table.setSpan(0, 1, 2, 1)
        for i, comp in enumerate(fitted_data.statistic):
            write(0, i*9+3, comp["name"])
            self.recorded_data_table.setSpan(0, i*9+3, 1, 8)
            write(1, i*9+3, "Fraction")
            write(1, i*9+4, "Mean (μm)")
            write(1, i*9+5, "Median (μm)")
            write(1, i*9+6, "Mode (μm)")
            write(1, i*9+7, "Variance")
            write(1, i*9+8, "Standard Deviation")
            write(1, i*9+9, "Skewness")
            write(1, i*9+10, "Kurtosis")

        row = self.recorded_data_count + 2
        write(row, 0, fitted_data.name)
        write(row, 1, fitted_data.mse, e=True)
        for i, comp in enumerate(fitted_data.statistic):
            write(row, i*9+3, comp.get("fraction"))
            write(row, i*9+4, comp.get("mean"))
            write(row, i*9+5, comp.get("median"))
            write(row, i*9+6, comp.get("mode"))
            write(row, i*9+7, comp.get("variance"))
            write(row, i*9+8, comp.get("standard_deviation"))
            write(row, i*9+9, comp.get("skewness"))
            write(row, i*9+10, comp.get("kurtosis"))

        self.recorded_data_count += 1


    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            self.__setattr__(setting, value)

    
    def on_data_item_clicked(self, row, column):
        self.sigDataSelected.emit(row)

    def on_focus_sample_changed(self, index):
        self.raw_data_table.setCurrentCell(index, 0)
        
    def show_recorded_table_menu(self, pos):
        self.recorded_table_menu.popup(QCursor.pos())

    def remove_recorded_selection(self):
        rows_to_remove = []
        for item in self.recorded_data_table.selectedRanges():
            assert item.topRow() == item.bottomRow()
            rows_to_remove.append(item.topRow())
        rows_to_remove.sort()
        offset = 0
        for row in rows_to_remove:
            self.recorded_data_table.removeRow(item.topRow()-offset)
            offset+=1
        
        self.recorded_data_count -= offset
        self.sigRemoveRecords.emit(rows_to_remove)

        