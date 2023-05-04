import logging
import os
from typing import *

from PySide6 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .. import QGRAIN_ROOT_PATH


def normal_color():
    return os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]


def highlight_color():
    return os.environ["QTMATERIAL_PRIMARYCOLOR"]


def background_color():
    return os.environ["QTMATERIAL_SECONDARYDARKCOLOR"]


def synchronize_theme():
    plt.rcParams["axes.facecolor"] = os.environ["QTMATERIAL_SECONDARYDARKCOLOR"]
    plt.rcParams["figure.facecolor"] = os.environ["QTMATERIAL_SECONDARYDARKCOLOR"]
    plt.rcParams["axes.edgecolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["axes.titlecolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["axes.labelcolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["grid.color"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["legend.labelcolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["xtick.color"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["xtick.labelcolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["ytick.color"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["ytick.labelcolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]

    plt.rcParams["savefig.dpi"] = 1200.0
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["figure.max_open_warning"] = False


def setup_matplotlib():
    plt.style.use(["science", "no-latex"])
    plt.set_cmap("tab10")
    plt.rcParams['axes.unicode_minus'] = False
    fonts_path = os.path.join(QGRAIN_ROOT_PATH, "assets")
    for font in os.listdir(fonts_path):
        if font[-4:] == ".ttf":
            font_manager.fontManager.addfont(os.path.join(fonts_path, font))
    plt.rcParams["font.family"] = "Source Han Sans CN"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.title_fontsize"] = 8
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    synchronize_theme()


class BaseChart(QtWidgets.QWidget):
    logger = logging.getLogger("QGrain.charts")

    def __init__(self, parent=None, figsize=(4, 3)):
        super().__init__(parent=parent)
        self._figure: plt.Figure = plt.figure(figsize=figsize)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self._canvas, 0, 0)
        self.menu = QtWidgets.QMenu(self._canvas)
        self.menu.setShortcutAutoRepeat(True)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)
        self.edit_figure_action = self.menu.addAction(self.tr("Edit Figure"))
        self.edit_figure_action.triggered.connect(lambda: self._toolbar.edit_parameters())
        self.configure_subplots_action = self.menu.addAction(self.tr("Configure Subplots"))
        self.configure_subplots_action.triggered.connect(lambda: self._toolbar.configure_subplots())
        self.save_figure_action = self.menu.addAction(self.tr("Save Figure"))
        self.save_figure_action.triggered.connect(lambda: self._toolbar.save_figure())
        self.normal_msg = QtWidgets.QMessageBox(parent=self)
        self._animation: Optional[FuncAnimation] = None

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

    def show_menu(self, pos: QtCore.QPoint):
        self.menu.popup(QtGui.QCursor.pos())

    def update_chart(self):
        pass

    def save_chart(self, filename: str, **kwargs):
        self._figure.savefig(filename, **kwargs)

    def save_animation(self, filename: str = None):
        if self._animation is None:
            return
        self._animation.pause()
        if filename is None:
            filename, format_str = QtWidgets.QFileDialog.getSaveFileName(
                self, self.tr("Choose a filename to save the animation of this SSU result"),
                ".", "MPEG-4 Video File (*.mp4);;Html Animation (*.html);;Graphics Interchange Format (*.gif)")
        if filename is None or filename == "":
            return
        progress_dialog = QtWidgets.QProgressDialog(
            self.tr("Saving Animation Frames..."), self.tr("Cancel"),
            0, 100, self)
        progress_dialog.setWindowTitle("QGrain")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)

        def callback(frame_number, total_frames):
            if progress_dialog.wasCanceled():
                raise StopIteration()
            progress_dialog.setValue(int(frame_number / total_frames * 100))
            QtCore.QCoreApplication.processEvents()

        try:
            if filename[-5:] == ".html":
                if not FFMpegWriter.isAvailable():
                    self.show_error(self.tr("FFMpeg is not installed."))
                else:
                    self.show_info(self.tr("Rendering the animation to a html5 video, it will take several minutes."))
                    html = self._animation.to_html5_video()
                    with open(filename, "w") as f:
                        f.write(html)
            elif filename[-4:] == ".gif":
                if not ImageMagickWriter.isAvailable():
                    self.show_error(self.tr("ImageMagick is not installed."))
                else:
                    self._animation.save(filename, writer="imagemagick", fps=10, progress_callback=callback)
            elif filename[-4:] == ".mp4":
                if not FFMpegWriter.isAvailable():
                    self.show_error(self.tr("FFMpeg is not installed."))
                else:
                    self._animation.save(filename, writer="ffmpeg", fps=10, progress_callback=callback)
        except StopIteration:
            self.logger.info("The saving task was canceled.")
        finally:
            progress_dialog.close()

    def retranslate(self):
        pass

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.StyleChange:
            setup_matplotlib()
            self._figure.clear()
            self.main_layout.removeWidget(self._canvas)
            self._canvas.setVisible(False)
            self._figure = plt.figure(figsize=self._figure.get_size_inches())
            self._canvas = FigureCanvas(self._figure)
            self._toolbar = NavigationToolbar(self._canvas, self)
            self.main_layout.addWidget(self._canvas, 0, 0)
            self.update_chart()
        elif event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()


from .BoxplotChart import BoxplotChart
from .FrequencyChart import FrequencyChart
from .Frequency3DChart import Frequency3DChart
from .FrequencyHeatmap import FrequencyHeatmap
from .FrequencyGroupChart import FrequencyGroupChart
from .CumulativeChart import CumulativeChart
from .diagrams import *
from .HierarchicalChart import HierarchicalChart
from .PCAResultChart import PCAResultChart
from .LossSeriesChart import LossSeriesChart
from .DistributionChart import DistributionChart
from .EMMAResultChart import EMMAResultChart
from .UDMResultChart import UDMResultChart
