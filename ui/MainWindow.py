import logging
import time
from typing import List

from PySide2.QtCore import (QCoreApplication, QEventLoop, QMutex, QSettings,
                            Qt, QThread, Signal)
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import (QAbstractItemView, QAction, QDockWidget,
                               QMainWindow, QMessageBox, QTableWidget,
                               QTableWidgetItem)

from models.SampleDataset import SampleDataset
from resolvers.GUIResolver import GUIResolver
from resolvers.MultiprocessingResolver import MultiProcessingResolver
from ui.AboutWindow import AboutWindow
from ui.ControlPanel import ControlPanel
from ui.DataManager import DataManager
from ui.DistributionCanvas import DistributionCanvas
from ui.LossCanvas import LossCanvas
from ui.PCAPanel import PCAPanel
from ui.RecordedDataTable import RecordedDataTable
from ui.SettingWindow import SettingWindow
from ui.TaskWindow import TaskWindow


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
    sigSetup = Signal()
    sigCleanup = Signal()
    logger = logging.getLogger("root.MainWindow")
    gui_logger = logging.getLogger("GUI")
    def __init__(self, isDark=True):
        super().__init__()
        if isDark:
            self.icon_folder = "./settings/icons/dark/"
        else:
            self.icon_folder = "./settings/icons/light/"
        self.init_ui(isDark)
        self.status_bar = self.statusBar()
        self.gui_fitting_thread = QThread()
        self.gui_resolver = GUIResolver()
        # disable multi-thread for debug
        self.gui_resolver.moveToThread(self.gui_fitting_thread)
        self.multiprocessing_fitting_thread = QThread()
        self.multiprocessing_resolver = MultiProcessingResolver()
        self.multiprocessing_resolver.moveToThread(self.multiprocessing_fitting_thread)
        self.data_manager = DataManager(self)
        self.connect_all()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.reset_dock_layout()

    def show_message(self, message):
        self.status_bar.showMessage(message)

    def init_ui(self, isDark: bool):
        # Menu
        self.file_menu = self.menuBar().addMenu(self.tr("File"))
        self.load_action = QAction(QIcon(self.icon_folder+"load.png"), self.tr("Load"), self)
        self.file_menu.addAction(self.load_action)
        self.save_action = QAction(QIcon(self.icon_folder+"save.png"), self.tr("Save"), self)
        self.file_menu.addAction(self.save_action)
        self.docks_menu = self.menuBar().addMenu(self.tr("Docks"))
        self.pca_panel_action = QAction(QIcon(self.icon_folder+"pca.png"), self.tr("PCA Panel"), self)
        self.docks_menu.addAction(self.pca_panel_action)
        self.loss_canvas_action = QAction(QIcon(self.icon_folder+"canvas.png"), self.tr("Loss Canvas"), self)
        self.docks_menu.addAction(self.loss_canvas_action)
        self.distribution_canvas_action = QAction(QIcon(self.icon_folder+"canvas.png"), self.tr("Distribution Canvas"), self)
        self.docks_menu.addAction(self.distribution_canvas_action)
        self.control_panel_action = QAction(QIcon(self.icon_folder+"control.png"), self.tr("Control Panel"), self)
        self.docks_menu.addAction(self.control_panel_action)
        self.raw_data_table_action = QAction(QIcon(self.icon_folder+"raw_data_table.png"), self.tr("Raw Data Table"), self)
        self.docks_menu.addAction(self.raw_data_table_action)
        self.recorded_data_table_action = QAction(QIcon(self.icon_folder+"recorded_data_table.png"), self.tr("Recorded Data Table"), self)
        self.docks_menu.addAction(self.recorded_data_table_action)
        self.reset_docks_actions = QAction(QIcon(self.icon_folder+"reset.png"), self.tr("Reset"), self)
        self.docks_menu.addAction(self.reset_docks_actions)
        self.settings_action = self.menuBar().addAction(self.tr("Settings"))
        self.about_action = self.menuBar().addAction(self.tr("About"))

        self.setDockNestingEnabled(True)
        # PCA Panel
        self.pca_panel_dock = QDockWidget(self.tr("PCA Panel"))
        self.pca_panel = PCAPanel(isDark=isDark)
        self.pca_panel_dock.setWidget(self.pca_panel)
        self.pca_panel_dock.setObjectName("PCAPanelDock")
        # Loss Canvas
        self.loss_canvas_dock = QDockWidget(self.tr("Loss Canvas"))
        self.loss_canvas = LossCanvas(isDark=isDark)
        self.loss_canvas_dock.setWidget(self.loss_canvas)
        self.loss_canvas_dock.setObjectName("LossCanvasDock")
        # Distribution Canvas
        self.distribution_canvas_dock = QDockWidget(self.tr("Distribution Canvas"))
        self.distribution_canvas = DistributionCanvas(isDark=isDark)
        self.distribution_canvas_dock.setWidget(self.distribution_canvas)
        self.distribution_canvas_dock.setObjectName("DistributionCanvasDock")

        # Control Panel
        self.control_panel_dock = QDockWidget(self.tr("Control Panel"))
        self.control_panel = ControlPanel()
        self.control_panel_dock.setWidget(self.control_panel)
        self.control_panel_dock.setObjectName("ControlPanelDock")

        # Raw Data Table
        self.raw_data_dock = QDockWidget(self.tr("Raw Data Table"))
        self.raw_data_table = QTableWidget()
        self.raw_data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.raw_data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.raw_data_table.setAlternatingRowColors(True)
        self.raw_data_dock.setWidget(self.raw_data_table)
        self.raw_data_dock.setObjectName("RawDataTableDock")

        # Recorded Data Table
        self.recorded_data_dock = QDockWidget(self.tr("Recorded Data Table"))
        self.recorded_data_table = RecordedDataTable()
        self.recorded_data_dock.setWidget(self.recorded_data_table)
        self.recorded_data_dock.setObjectName("RecordedDataTableDock")

        self.settings_window = SettingWindow(self)
        self.about_window = AboutWindow(self)
        self.task_window = TaskWindow(self)

    def connect_all(self):
        self.control_panel.sigDistributionTypeChanged.connect(self.gui_resolver.on_distribution_type_changed)
        self.control_panel.sigDistributionTypeChanged.connect(self.multiprocessing_resolver.on_distribution_type_changed)
        self.control_panel.sigComponentNumberChanged.connect(self.gui_resolver.on_component_number_changed)
        self.control_panel.sigComponentNumberChanged.connect(self.multiprocessing_resolver.on_component_number_changed)
        self.control_panel.sigComponentNumberChanged.connect(self.distribution_canvas.on_component_number_changed)
        self.control_panel.sigFocusSampleChanged.connect(self.data_manager.on_focus_sample_changed)
        self.control_panel.sigFocusSampleChanged.connect(self.on_focus_sample_changed)
        self.control_panel.sigGUIResolverSettingsChanged.connect(self.gui_resolver.on_settings_changed)
        self.control_panel.sigGUIResolverFittingStarted.connect(self.gui_resolver.try_fit)

        self.control_panel.sigDataSettingsChanged.connect(self.data_manager.on_settings_changed)
        self.control_panel.sigGUIResolverFittingCanceled.connect(self.on_task_canceled)
        self.control_panel.sigMultiProcessingFittingStarted.connect(self.multiprocessing_resolver.execute_tasks)
        # Connect directly
        self.control_panel.record_button.clicked.connect(self.data_manager.record_current_data)

        self.distribution_canvas.sigExpectedMeanValueChanged.connect(self.gui_resolver.on_excepted_mean_value_changed)

        self.data_manager.sigDataLoaded.connect(self.on_data_loaded)
        self.data_manager.sigDataLoaded.connect(self.control_panel.on_data_loaded)
        self.data_manager.sigDataLoaded.connect(self.multiprocessing_resolver.on_data_loaded)
        self.data_manager.sigDataLoaded.connect(self.pca_panel.on_data_loaded)
        self.data_manager.sigTargetDataChanged.connect(self.gui_resolver.on_target_data_changed)
        self.data_manager.sigTargetDataChanged.connect(self.distribution_canvas.on_target_data_changed)
        self.data_manager.sigDataRecorded.connect(self.recorded_data_table.on_data_recorded)

        self.gui_resolver.sigFittingStarted.connect(self.control_panel.on_fitting_started)
        self.gui_resolver.sigFittingFinished.connect(self.control_panel.on_fitting_finished)
        self.gui_resolver.sigFittingStarted.connect(self.loss_canvas.on_fitting_started)
        self.gui_resolver.sigFittingFinished.connect(self.loss_canvas.on_fitting_finished)
        self.gui_resolver.sigFittingEpochSucceeded.connect(self.control_panel.on_fitting_epoch_suceeded)
        self.gui_resolver.sigFittingEpochSucceeded.connect(self.distribution_canvas.on_fitting_epoch_suceeded)
        self.gui_resolver.sigFittingEpochSucceeded.connect(self.data_manager.on_fitting_epoch_suceeded)
        self.gui_resolver.sigSingleIterationFinished.connect(self.loss_canvas.on_single_iteration_finished)
        self.gui_resolver.sigSingleIterationFinished.connect(self.distribution_canvas.on_single_iteration_finished)
        self.gui_resolver.sigFittingFailed.connect(self.control_panel.on_fitting_failed)

        self.multiprocessing_resolver.sigTaskInitialized.connect(self.task_window.on_task_initialized)
        self.multiprocessing_resolver.sigTaskStateUpdated.connect(self.task_window.on_task_state_updated)
        self.multiprocessing_resolver.sigTaskFinished.connect(self.task_window.on_task_finished)
        self.multiprocessing_resolver.sigTaskFinished.connect(self.data_manager.on_multiprocessing_task_finished)

        self.raw_data_table.cellClicked.connect(self.on_data_item_clicked)
        self.recorded_data_table.sigRemoveRecords.connect(self.data_manager.remove_data)

        # Dock menu actions
        self.pca_panel_action.triggered.connect(self.show_pca_panel_dock)
        self.loss_canvas_action.triggered.connect(self.show_loss_canvas_dock)
        self.distribution_canvas_action.triggered.connect(self.show_distribution_canvas_dock)
        self.control_panel_action.triggered.connect(self.show_control_panel_dock)
        self.raw_data_table_action.triggered.connect(self.show_raw_data_dock)
        self.recorded_data_table_action.triggered.connect(self.show_recorded_data_dock)
        self.reset_docks_actions.triggered.connect(self.reset_dock_layout)
        self.settings_action.triggered.connect(self.settings_window.show)
        self.about_action.triggered.connect(self.about_window.show)

        self.load_action.triggered.connect(self.data_manager.load_data)
        self.save_action.triggered.connect(self.data_manager.save_data)
        self.sigDataSelected.connect(self.control_panel.on_data_selected)

        self.sigSetup.connect(self.control_panel.setup_all)
        self.sigSetup.connect(self.data_manager.setup_all)
        self.sigSetup.connect(self.settings_window.setup_all)
        self.sigSetup.connect(self.multiprocessing_resolver.setup_all)
        self.sigCleanup.connect(self.data_manager.cleanup_all)
        self.sigCleanup.connect(self.multiprocessing_resolver.cleanup_all)

        self.settings_window.data_setting.sigDataSettingChanged.connect(self.data_manager.on_settings_changed)
        self.settings_window.algorithm_setting.sigAlgorithmSettingChanged.connect(self.gui_resolver.on_algorithm_settings_changed)
        self.settings_window.algorithm_setting.sigAlgorithmSettingChanged.connect(self.multiprocessing_resolver.on_algorithm_settings_changed)

    def reset_dock_layout(self):
        self.pca_panel_dock.show()
        self.loss_canvas_dock.show()
        self.distribution_canvas_dock.show()
        self.raw_data_dock.show()
        self.recorded_data_dock.show()
        self.control_panel_dock.show()

        self.pca_panel_dock.setFloating(False)
        self.distribution_canvas_dock.setFloating(False)
        self.loss_canvas_dock.setFloating(False)
        self.control_panel_dock.setFloating(False)
        self.raw_data_dock.setFloating(False)
        self.recorded_data_dock.setFloating(False)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.pca_panel_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.raw_data_dock)
        self.splitDockWidget(self.raw_data_dock, self.recorded_data_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self.pca_panel_dock, self.control_panel_dock, Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.pca_panel_dock, self.loss_canvas_dock)
        self.tabifyDockWidget(self.pca_panel_dock, self.distribution_canvas_dock)

        self.resizeDocks((self.pca_panel_dock, self.control_panel_dock), (self.height()*0.6, self.height()*0.4), Qt.Orientation.Vertical)
        self.resizeDocks((self.pca_panel_dock, self.control_panel_dock, self.raw_data_dock, self.recorded_data_dock), (self.width()*0.5, self.width()*0.5, self.width()*0.25, self.width()*0.25), Qt.Orientation.Horizontal)

    def show_pca_panel_dock(self):
        self.pca_panel_dock.show()

    def show_loss_canvas_dock(self):
        self.loss_canvas_dock.show()

    def show_distribution_canvas_dock(self):
        self.distribution_canvas_dock.show()

    def show_control_panel_dock(self):
        self.control_panel_dock.show()

    def show_raw_data_dock(self):
        self.raw_data_dock.show()

    def show_recorded_data_dock(self):
        self.recorded_data_dock.show()

    def on_data_loaded(self, dataset: SampleDataset):
        nrows = dataset.data_count
        ncols = len(dataset.classes)
        self.raw_data_table.setRowCount(nrows)
        self.raw_data_table.setColumnCount(ncols)
        self.raw_data_table.setHorizontalHeaderLabels(["{0:.2f}".format(class_value) for class_value in dataset.classes])
        self.raw_data_table.setVerticalHeaderLabels([str(sample.name) for sample in dataset.samples])
        for i, sample in enumerate(dataset.samples):
            QCoreApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            for j, value in enumerate(sample.distribution):
                item = QTableWidgetItem("{0:.4f}".format(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.raw_data_table.setItem(i, j, item)

        # resize to tight layout
        self.raw_data_table.resizeColumnsToContents()
        self.logger.info("Data was loaded, and has been update to the table.")

    def on_data_item_clicked(self, row, column):
        self.sigDataSelected.emit(row)
        self.logger.debug("The item at [%d] row in raw data table was selected.", row)

    def on_focus_sample_changed(self, index):
        self.raw_data_table.setCurrentCell(index, 0)

    def on_task_canceled(self):
        self.gui_resolver.cancel()

    def setup_all(self):
        # must call this again to make the layout as expected
        self.reset_dock_layout()
        # emit signal to let other object to setup
        self.gui_fitting_thread.start()
        self.multiprocessing_fitting_thread.start()
        # emit after the threads are started
        self.sigSetup.emit()

        settings = QSettings("./settings/ui.ini", QSettings.IniFormat)
        settings.beginGroup("MainWindow")
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        windowState = settings.value("windowState")
        if windowState is not None:
            self.restoreState(windowState)
        settings.endGroup()

    def closeEvent(self, e):
        # TODO: add task running check
        self.sigCleanup.emit()
        self.hide()

        settings = QSettings("./settings/ui.ini", QSettings.IniFormat)
        settings.beginGroup("MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.endGroup()

        time.sleep(2)
        self.gui_fitting_thread.terminate()
        self.multiprocessing_fitting_thread.terminate()
        e.accept()
