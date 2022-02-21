import logging
import os
import sys
import typing
from logging.handlers import TimedRotatingFileHandler

import matplotlib as mpl
import matplotlib.pyplot as plt
from PySide6 import QtCore, QtGui, QtWidgets
from qt_material import apply_stylesheet, list_themes

from .. import QGRAIN_ROOT_PATH, QGRAIN_VERSION
from ..chart import setup_matplotlib
from ..io import save_pca, save_statistic
from ..model import GrainSizeDataset
from .AboutDialog import AboutDialog
from .ClusteringAnalyzer import ClusteringAnalyzer
from .DatasetGenerator import DatasetGenerator
from .EMMASettingDialog import EMMASettingDialog
from .GrainSizeDatasetViewer import GrainSizeDatasetViewer
from .LoadDatasetDialog import LoadDatasetDialog
from .LogDialog import *
from .PCAAnalyzer import PCAAnalyzer
from .SSUAnalyzer import SSUAnalyzer
from .SSUSettingDialog import SSUSettingDialog
from .ParameterEditor import ParameterEditor

class MainWindow(QtWidgets.QMainWindow):
    logger = logging.getLogger("QGrain")
    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.tr("QGrain"))
        self.__dataset = GrainSizeDataset() # type: GrainSizeDataset
        self.current_translator = None # type: QtCore.QTranslator
        self.ssu_setting_dialog = SSUSettingDialog()
        self.emma_setting_dialog = EMMASettingDialog()
        self.init_ui()
        self.load_dataset_dialog = LoadDatasetDialog(self)
        self.load_dataset_dialog.dataset_loaded.connect(self.on_dataset_loaded)
        self.load_dataset_dialog.dataset_loaded.connect(self.dataset_viewer.on_dataset_loaded)
        self.load_dataset_dialog.dataset_loaded.connect(self.pca_analyzer.on_dataset_loaded)
        self.load_dataset_dialog.dataset_loaded.connect(self.clustering_analyzer.on_dataset_loaded)
        self.load_dataset_dialog.dataset_loaded.connect(self.ssu_analyzer.on_dataset_loaded)
        self.log_dialog = LogDialog(self)
        self.about_dialog = AboutDialog(self)
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.close_msg = QtWidgets.QMessageBox(self)
        self.close_msg.setWindowTitle(self.tr("Warning"))
        self.close_msg.setText(self.tr("Closing this window will terminate all running tasks, are you sure to close it?"))
        self.close_msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        self.close_msg.setDefaultButton(QtWidgets.QMessageBox.No)

        # load the artificial dataset to show the functions of all modules
        dataset = self.dataset_generator.get_random_dataset(100)
        self.load_dataset_dialog.dataset_loaded.emit(dataset.dataset_to_fit)
        self.ssu_analyzer.on_try_fit_clicked()

    def init_ui(self):
        self.tab_widget = QtWidgets.QTabWidget(self)
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.West)
        self.setCentralWidget(self.tab_widget)
        self.dataset_generator = DatasetGenerator()
        self.tab_widget.addTab(self.dataset_generator, self.tr("Dataset Generator"))
        self.dataset_viewer = GrainSizeDatasetViewer()
        self.tab_widget.addTab(self.dataset_viewer, self.tr("Dataset Viewer"))
        self.pca_analyzer = PCAAnalyzer()
        self.tab_widget.addTab(self.pca_analyzer, self.tr("PCA Analyzer"))
        self.clustering_analyzer = ClusteringAnalyzer()
        self.tab_widget.addTab(self.clustering_analyzer, self.tr("Clustering Analyzer"))
        # self.emma_panel = QtWidgets.QWidget()
        # self.tab_widget.addTab(self.emma_panel, self.tr("EMMA Analyzer"))
        self.ssu_analyzer = SSUAnalyzer(setting_dialog=self.ssu_setting_dialog)
        self.tab_widget.addTab(self.ssu_analyzer, self.tr("SSU Analyzer"))
        # self.udm_panel = QtWidgets.QWidget()
        # self.tab_widget.addTab(self.udm_panel, self.tr("UDM Analyzer"))

        self.init_menus()

    def init_menus(self):
        # Open
        self.open_menu = self.menuBar().addMenu(self.tr("Open")) # type: QtWidgets.QMenu
        self.open_dataset_action = self.open_menu.addAction(self.tr("Grain Size Dataset"))
        self.open_dataset_action.triggered.connect(lambda: self.load_dataset_dialog.show())
        # Save
        self.save_menu = self.menuBar().addMenu(self.tr("Save")) # type: QtWidgets.QMenu
        self.save_statistic_action = self.save_menu.addAction(self.tr("Statistic Result"))
        self.save_statistic_action.triggered.connect(self.on_save_statistic_clicked)
        self.save_pca_action = self.save_menu.addAction(self.tr("PCA Result"))
        self.save_pca_action.triggered.connect(self.on_save_pca_clicked)
        self.save_clustering_action = self.save_menu.addAction(self.tr("Clustering Result"))
        self.save_clustering_action.triggered.connect(self.clustering_analyzer.on_save_clicked)
        # Config
        self.config_menu = self.menuBar().addMenu(self.tr("Config")) # type: QtWidgets.QMenu
        self.config_ssu_action = self.config_menu.addAction(self.tr("SSU Settings"))
        self.config_ssu_action.triggered.connect(lambda: self.ssu_setting_dialog.show())
        self.config_emma_action = self.config_menu.addAction(self.tr("EMMA Settings"))
        self.config_emma_action.triggered.connect(lambda: self.emma_setting_dialog.show())

        # Language
        self.language_menu = self.menuBar().addMenu(self.tr("Language")) # type: QtWidgets.QMenu
        self.language_group = QtGui.QActionGroup(self.language_menu)
        self.language_group.setExclusive(True)
        self.language_actions = [] # type: list[QtGui.QAction]
        for key, name in self.supported_languages:
            action = self.language_group.addAction(name)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, language=key: self.switch_language(language))
            self.language_menu.addAction(action)
            self.language_actions.append(action)
        self.language_actions[0].setChecked(True)
        # Theme
        self.theme_menu = self.menuBar().addMenu(self.tr("Theme")) # type: QtWidgets.QMenu
        self.theme_group = QtGui.QActionGroup(self.theme_menu)
        self.theme_group.setExclusive(True)
        self.theme_actions = []
        for theme in list_themes():
            action = self.theme_group.addAction(theme)
            action.setCheckable(True)
            app = QtCore.QCoreApplication.instance()
            invert_secondary = theme.startswith("light")
            action.triggered.connect(lambda checked=False, theme=theme, invert_secondary=invert_secondary: apply_stylesheet(app, theme=theme, invert_secondary=invert_secondary, extra=EXTRA))
            self.theme_menu.addAction(action)
            self.theme_actions.append(action)
            if theme == "dark_cyan.xml":
                action.setChecked(True)
        # Log
        self.log_action = QtGui.QAction(self.tr("Log"))
        self.log_action.triggered.connect(lambda: self.log_dialog.show())
        self.menuBar().addAction(self.log_action)
        # About
        self.about_action = QtGui.QAction(self.tr("About"))
        self.about_action.triggered.connect(lambda: self.about_dialog.show())
        self.menuBar().addAction(self.about_action)

    @property
    def supported_languages(self) -> typing.List[typing.Tuple[str, str]]:
        languages = [("en", "English"),
                     ("zh_CN", "简体中文")]
        return languages

    @property
    def language(self) -> str:
        for i, language_action in enumerate(self.language_actions):
            if language_action.isChecked():
                key, name = self.supported_languages[i]
                return key

    @property
    def theme(self) -> str:
        for i, theme_action in enumerate(self.theme_actions):
            if theme_action.isChecked():
                theme = list_themes()[i]
                return theme

    def show_message(self, title: str, message: str):
        self.normal_msg.setWindowTitle(title)
        self.normal_msg.setText(message)
        self.normal_msg.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    def on_dataset_loaded(self, dataset: GrainSizeDataset):
        if dataset is None or not dataset.has_sample:
            return
        self.__dataset = dataset

    def on_save_statistic_clicked(self):
        if not self.__dataset.has_sample:
            self.show_warning(self.tr("Dataset has not been loaded."))
            return
        filename, _ = self.file_dialog.getSaveFileName(
            self, self.tr("Save statistic result"),
            None, "Microsoft Excel (*.xlsx)")
        if filename is None or filename == "":
            return
        try:
            progress_dialog = QtWidgets.QProgressDialog(
                self.tr("Saving statistic result..."), self.tr("Cancel"),
                0, 100, self)
            progress_dialog.setWindowTitle(self.tr("QGrain"))
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            def callback(progress: float):
                if progress_dialog.wasCanceled():
                    raise StopIteration()
                progress_dialog.setValue(int(progress*100))
                QtCore.QCoreApplication.processEvents()
            save_statistic(self.__dataset, filename, progress_callback=callback)
            progress_dialog.setValue(100)
            self.logger.info("Statistic result of this dataset has been saved to the Excel file.")
        except Exception as e:
            self.logger.exception("An unknown exception was raised. Please check the logs for more details.", stack_info=True)
            self.show_error(self.tr("An unknown exception was raised. Please check the logs for more details."))

    def on_save_pca_clicked(self):
        if not self.__dataset.has_sample:
            self.show_warning(self.tr("Dataset has not been loaded."))
            return
        filename, _ = self.file_dialog.getSaveFileName(
            None, self.tr("Save PCA result"),
            None, "Microsoft Excel (*.xlsx)")
        if filename is None or filename == "":
            return
        try:
            progress_dialog = QtWidgets.QProgressDialog(
                self.tr("Saving PCA result..."), self.tr("Cancel"),
                0, 100, self)
            progress_dialog.setWindowTitle(self.tr("QGrain"))
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            def callback(progress: float):
                if progress_dialog.wasCanceled():
                    raise StopIteration()
                progress_dialog.setValue(int(progress*100))
                QtCore.QCoreApplication.processEvents()
            save_pca(self.__dataset, filename, progress_callback=callback)
            progress_dialog.setValue(100)
            self.logger.info("PCA result of this dataset has been saved to the Excel file.")
        except Exception as e:
            self.logger.exception("An unknown exception was raised. Please check the logs for more details.", stack_info=True)
            self.show_error(self.tr("An unknown exception was raised. Please check the logs for more details."))

    def switch_language(self, language: str):
        app = QtWidgets.QApplication.instance()
        if self.current_translator is not None:
            app.removeTranslator(self.current_translator)
        translator = QtCore.QTranslator(app)
        translator.load(os.path.join(QGRAIN_ROOT_PATH, "assets", language))
        app.installTranslator(translator)
        self.current_translator = translator

    def retranslate(self):
        self.setWindowTitle(self.tr("QGrain"))
        self.close_msg.setWindowTitle(self.tr("Warning"))
        self.close_msg.setText(self.tr("Closing this window will terminate all running tasks, are you sure to close it?"))
        self.open_menu.setTitle((self.tr("Open")))
        self.save_menu.setTitle(self.tr("Save"))
        self.open_dataset_action.setText(self.tr("Grain Size Dataset"))
        self.save_statistic_action.setText(self.tr("Statistic Result"))
        self.save_pca_action.setText(self.tr("PCA Result"))
        self.save_clustering_action.setText(self.tr("Clustering Result"))
        self.language_menu.setTitle(self.tr("Language"))
        self.theme_menu.setTitle(self.tr("Theme"))
        self.log_action.setText(self.tr("Log"))
        self.about_action.setText(self.tr("About"))
        self.tab_widget.setTabText(0, self.tr("Dataset Generator"))
        self.tab_widget.setTabText(1, self.tr("Dataset Viewer"))

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def closeEvent(self, event: QtGui.QCloseEvent):
        res = self.close_msg.exec_()
        if res == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


EXTRA = {'font_family': 'Roboto,Arial,Helvetica,Tahoma,Verdana,Microsoft YaHei UI,SimSum'}

def getdirsize(dir):
    size = 0
    for root, _, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size

def create_necessary_folders():
    necessary_folders = (
        os.path.join(os.path.expanduser("~"), "QGrain"),
        os.path.join(os.path.expanduser("~"), "QGrain", "logs"))
    for folder in necessary_folders:
        os.makedirs(folder, exist_ok=True)

def setup_language(app: QtWidgets.QApplication, language: str):
    trans = QtCore.QTranslator(app)
    trans.load(os.path.join(QGRAIN_ROOT_PATH, "assets", language))
    app.installTranslator(trans)

def setup_logging(main: MainWindow):
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler = TimedRotatingFileHandler(os.path.join(os.path.expanduser("~"), "QGrain", "logs", "qgrain.log"), when="D", backupCount=8, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(format_str))
    status_bar_handler = StatusBarLogHandler(main.statusBar(), level=logging.INFO)
    status_bar_handler.setFormatter(logging.Formatter(format_str))
    gui_handler = GUILogHandler(main.log_dialog, level=logging.DEBUG)
    gui_handler.setFormatter(logging.Formatter(format_str))
    logging.basicConfig(level=logging.DEBUG, format=format_str)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(status_bar_handler)
    logging.getLogger().addHandler(gui_handler)
    mpl.set_loglevel("error")

def setup_app():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    # for filename in os.listdir(os.path.join(QGRAIN_ROOT_PATH, "assets")):
    #     if filename.endswith(".ttf"):
    #         font_path = os.path.join(QGRAIN_ROOT_PATH, "assets", filename)
    #         font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
    #         font_family = QtGui.QFontDatabase.applicationFontFamilies(font_id)
    splash = QtWidgets.QSplashScreen()
    pixmap = QtGui.QPixmap(os.path.join(QGRAIN_ROOT_PATH, "assets", "icon.png"))
    pixmap.setDevicePixelRatio(1.0)
    splash.setPixmap(pixmap)
    splash.show()

    create_necessary_folders()
    app.setWindowIcon(QtGui.QIcon(pixmap))
    # app.setApplicationDisplayName(f"QGrain ({QGRAIN_VERSION})")
    app.setApplicationVersion(QGRAIN_VERSION)
    from qt_material import apply_stylesheet
    apply_stylesheet(app, theme='dark_cyan.xml', invert_secondary=False, extra=EXTRA)
    setup_matplotlib()
    setup_language(app, "en")
    return app, splash

def qgrain_app():
    app, splash = setup_app()
    main = MainWindow()
    setup_logging(main)
    # main.move(main.screen().geometry().center() - main.frameGeometry().center())
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
