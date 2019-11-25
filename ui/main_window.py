import logging
import os
import sys
import time
from queue import Queue

import numpy as np
import pyqtgraph as pg
import xlrd
from pyqtgraph.dockarea import Dock, DockArea
from PySide2.QtCore import QMutex, Qt, QThread, QTimer, Signal
from PySide2.QtGui import QCursor, QFont, QIcon, QAction
from PySide2.QtWidgets import (QAbstractItemView, QApplication, QGridLayout, QMessageBox,
                               QLabel, QMainWindow, QMenu, QPushButton, QToolBar,
                               QSizePolicy, QSplitter, QTableWidget,
                               QTableWidgetItem, QWidget)

from data import DataManager, FittedData, GrainSizeData
from resolvers import Resolver
from ui import ControlPanel, FittingCanvas
from ui.setting_window import SettingWindow
from ui.about_window import AboutWindow


class GUILogHandler(logging.Handler):
    def __init__(self, target_widget):
        logging.Handler.__init__(self)
        self.target = target_widget
        self.__mutex = QMutex()

    def emit(self, record):
        self.__mutex.lock()
        if record.levelno < self.level:
            return
        self.target.show_message(record.msg % record.args)
        self.__mutex.unlock()


class MainWindow(QMainWindow):
    sigDataSelected = Signal(int) 
    sigRemoveRecords = Signal(list)
    TABLE_HEADER_ROWS = 2
    COPONENT_ITEMS = 12
    logger = logging.getLogger("root.MainWindow")
    gui_logger = logging.getLogger("GUI")
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
        self.msg_box = QMessageBox()
        self.msg_box.setWindowFlags(Qt.Drawer)

    def show_message(self, message):
        self.status_bar.showMessage(message)

    def init_ui(self):
        self.dock_area = DockArea(self)
        self.setCentralWidget(self.dock_area)
        # Menu
        self.file_menu = self.menuBar().addMenu(self.tr("File"))
        self.load_action = QAction(QIcon("./settings/icons/open.png"), self.tr("Load"), self)
        self.file_menu.addAction(self.load_action)
        self.save_action = QAction(QIcon("./settings/icons/save.png"), self.tr("Save"), self)
        self.file_menu.addAction(self.save_action)
        self.docks_menu = self.menuBar().addMenu(self.tr("Docks"))
        self.canvas_action = QAction(QIcon("./settings/icons/canvas.png"), self.tr("Canvas"), self)
        self.docks_menu.addAction(self.canvas_action)
        self.control_panel_action = QAction(QIcon("./settings/icons/control.png"), self.tr("Control Panel"), self)
        self.docks_menu.addAction(self.control_panel_action)
        self.raw_data_table_action = QAction(QIcon("./settings/icons/raw_table.png"), self.tr("Raw Table"), self)
        self.docks_menu.addAction(self.raw_data_table_action)
        self.recorded_data_table_action = QAction(QIcon("./settings/icons/recorded_table.png"), self.tr("Recorded Table"), self)
        self.docks_menu.addAction(self.recorded_data_table_action)
        self.reset_docks_actions = QAction(QIcon("./settings/icons/reset.png"), self.tr("Reset"), self)
        self.docks_menu.addAction(self.reset_docks_actions)
        self.settings_action = self.menuBar().addAction(self.tr("Settings"))
        self.about_action = self.menuBar().addAction(self.tr("About"))
        
        # Canvas
        self.canvas_dock = Dock(self.tr("Canvas"), size=(200, 300), closable=True)
        self.dock_area.addDock(self.canvas_dock)
        self.canvas = FittingCanvas()
        self.canvas_dock.addWidget(self.canvas)

        # Control Panel
        self.control_panel_dock = Dock(self.tr("Control Panel"), size=(200, 100), closable=True)
        self.control_panel_dock.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.dock_area.addDock(self.control_panel_dock)
        self.control_panel = ControlPanel()
        self.control_panel_dock.addWidget(self.control_panel)

        # Raw Data Table
        self.raw_data_dock = Dock(self.tr("Raw Data Table"), size=(300, 400), closable=True)
        self.dock_area.addDock(self.raw_data_dock)
        self.raw_data_table = QTableWidget()
        self.raw_data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.raw_data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.raw_data_table.setAlternatingRowColors(True)
        self.raw_data_dock.addWidget(self.raw_data_table)

        # Recorded Data Table
        self.recorded_data_dock = Dock(self.tr("Recorded Data Table"), size=(300, 400), closable=True)
        self.dock_area.addDock(self.recorded_data_dock)
        self.recorded_data_table = QTableWidget()
        self.recorded_data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.recorded_data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.recorded_data_table.setAlternatingRowColors(True)
        self.recorded_data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.recorded_data_dock.addWidget(self.recorded_data_table)
        self.recorded_table_menu = QMenu(self.recorded_data_table)
        self.recorded_table_remove_action = self.recorded_table_menu.addAction(self.tr("Remove"))
        
        self.settings_window = SettingWindow()
        self.about_window = AboutWindow()
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
        # Connect directly
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
        self.resolver.sigWidgetsEnable.connect(self.control_panel.on_widgets_enable_changed)
        
        # Dock menu actions
        self.load_action.triggered.connect(self.data_manager.load_data)
        self.save_action.triggered.connect(self.data_manager.save_data)
        self.canvas_action.triggered.connect(self.show_canvas_dock)
        self.control_panel_action.triggered.connect(self.show_control_panel_dock)
        self.raw_data_table_action.triggered.connect(self.show_raw_data_dock)
        self.recorded_data_table_action.triggered.connect(self.show_recorded_data_dock)
        self.reset_docks_actions.triggered.connect(self.reset_dock_layout)
        self.settings_action.triggered.connect(self.settings_window.show)
        self.about_action.triggered.connect(self.about_window.show)
        
        self.canvas.sigWidgetsEnable.connect(self.control_panel.on_widgets_enable_changed)
        self.raw_data_table.cellClicked.connect(self.on_data_item_clicked)
        self.recorded_table_remove_action.triggered.connect(self.remove_recorded_selection)
        self.recorded_data_table.customContextMenuRequested.connect(self.show_recorded_table_menu)
        self.sigDataSelected.connect(self.control_panel.on_data_selected)
        self.sigRemoveRecords.connect(self.data_manager.remove_data)

    def reset_dock_layout(self):
        self.dock_area.moveDock(self.canvas_dock, "left", None)
        self.dock_area.moveDock(self.raw_data_dock, "right", self.canvas_dock)
        self.dock_area.moveDock(self.recorded_data_dock, "right", self.raw_data_dock)
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
        self.logger.debug("Data was loaded, and has been update to the table.")

    def on_data_recorded(self, fitted_data: FittedData):
        # the 2 additional rows are headers
        if self.recorded_data_count + self.TABLE_HEADER_ROWS >= self.recorded_data_table.rowCount():
            self.recorded_data_table.setRowCount(self.recorded_data_table.rowCount() + 50)
        ncomp = len(fitted_data.statistic)
        column_span = self.COPONENT_ITEMS + 1
        if ncomp*self.COPONENT_ITEMS+self.TABLE_HEADER_ROWS > self.recorded_data_table.columnCount():
            self.recorded_data_table.setColumnCount(ncomp*column_span+2)

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
        try:
            # Write Header
            write(0, 0, self.tr("Sample Name"))
            self.recorded_data_table.setSpan(0, 0, self.TABLE_HEADER_ROWS, 1)
            write(0, 1, self.tr("Mean Squared Error"))
            self.recorded_data_table.setSpan(0, 1, self.TABLE_HEADER_ROWS, 1)
            for i, comp in enumerate(fitted_data.statistic):
                write(0, i*column_span+3, comp["name"])
                self.recorded_data_table.setSpan(0, i*column_span+3, 1, self.COPONENT_ITEMS)
                write(1, i*column_span+3, self.tr("Fraction"))
                write(1, i*column_span+4, self.tr("Mean (μm)"))
                write(1, i*column_span+5, self.tr("Median (μm)"))
                write(1, i*column_span+6, self.tr("Mode (μm)"))
                write(1, i*column_span+7, self.tr("Variance"))
                write(1, i*column_span+8, self.tr("Standard Deviation"))
                write(1, i*column_span+9, self.tr("Skewness"))
                write(1, i*column_span+10, self.tr("Kurtosis"))
                write(1, i*column_span+11, self.tr("Beta"))
                write(1, i*column_span+12, self.tr("Eta"))
                write(1, i*column_span+13, self.tr("Location"))
                write(1, i*column_span+14, self.tr("X Offset"))

            row = self.recorded_data_count + self.TABLE_HEADER_ROWS
            write(row, 0, fitted_data.name)
            write(row, 1, fitted_data.mse, e=True)
            for i, comp in enumerate(fitted_data.statistic):
                write(row, i*column_span+3, comp.get("fraction"))
                write(row, i*column_span+4, comp.get("mean"))
                write(row, i*column_span+5, comp.get("median"))
                write(row, i*column_span+6, comp.get("mode"))
                write(row, i*column_span+7, comp.get("variance"))
                write(row, i*column_span+8, comp.get("standard_deviation"))
                write(row, i*column_span+9, comp.get("skewness"))
                write(row, i*column_span+10, comp.get("kurtosis"))
                write(row, i*column_span+11, comp.get("beta"))
                write(row, i*column_span+12, comp.get("eta"))
                write(row, i*column_span+13, comp.get("loc"))
                write(row, i*column_span+14, comp.get("x_offset"))
            self.recorded_data_count += 1
            self.logger.debug("Fitted data of [%s] has been wrote to recorded data table.", fitted_data.name)
        except Exception:
            self.logger.exception("Unknown exception occurred when writing fitted data to the table widget.")
            self.gui_logger.exception(self.tr("Unknown exception occurred when writing fitted data to the table widget."))
            # remove
            self.sigRemoveRecords(self.recorded_data_count)

    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            setattr(self, setting, value)
            self.logger.debug("Setting [%s] have been changed to [%s].", setting, value)

    def on_data_item_clicked(self, row, column):
        self.sigDataSelected.emit(row)
        self.logger.debug("The item at [%d] row in raw data table was selected.", row)

    def on_focus_sample_changed(self, index):
        self.raw_data_table.setCurrentCell(index, 0)
        
    def show_recorded_table_menu(self, pos):
        self.recorded_table_menu.popup(QCursor.pos())

    def remove_recorded_selection(self):
        rows_to_remove = set()
        # The behaviour of `selectedRanges` differs when clicking in table or at the edge
        # When clicking in table it returns multi ranges (each row)
        # When clicking at the edge it returns a single range
        for item in self.recorded_data_table.selectedRanges():
            for i in range(item.topRow(), min(self.recorded_data_count+self.TABLE_HEADER_ROWS, item.bottomRow()+1)):
                rows_to_remove.add(i)
        rows_to_remove = list(rows_to_remove)
        rows_to_remove.sort()
        offset = 0
        for row in rows_to_remove:
            self.recorded_data_table.removeRow(row-offset)
            offset += 1
        self.recorded_data_count -= offset
        records_to_remove = [row-self.TABLE_HEADER_ROWS for row in rows_to_remove]
        self.sigRemoveRecords.emit(records_to_remove)
        self.logger.debug("The rows of recorded data will be remove are: [%s].", records_to_remove)
