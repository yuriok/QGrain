
import math
import sys
from typing import Optional, Union

import numpy as np
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QLocale, QPointF, QRectF, QSizeF, Qt, QTimer, QSize, QStandardPaths
from PySide2.QtGui import (QBrush, QColor, QCursor, QFont, QImage, QPainter,QIntValidator,QDoubleValidator, QPixmap,
                           QPen)
from PySide2.QtSvg import QSvgGenerator
from PySide2.QtWidgets import (QAction, QApplication, QColorDialog, QDialog, QLabel, QComboBox,QLineEdit,QFileDialog, QMessageBox,
                               QGraphicsItem, QGraphicsScene,
                               QGraphicsSceneDragDropEvent,
                               QGraphicsSceneHoverEvent,
                               QGraphicsSceneMouseEvent, QGraphicsView,
                               QGridLayout, QMainWindow, QPushButton,
                               QSizePolicy, QStyleOptionGraphicsItem, QWidget)



class ChartExportDialog(QDialog):
    def __init__(self, canvas):
        super().__init__(parent=canvas)
        self.canvas = canvas
        self.file_dialog = QFileDialog(self)
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.init_ui()
        self.setAttribute(Qt.WA_StyledBackground, True)

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(400, 300)
        self.main_layout.addWidget(self.preview_label, 0, 0, 1, 4)
        self.int_validator = QIntValidator()
        self.int_validator.setRange(100, 10000)
        self.double_validator = QDoubleValidator()
        self.double_validator.setRange(1.00, 10.00)
        self.figure_width_label = QLabel(self.tr("Width"))
        self.main_layout.addWidget(self.figure_width_label, 1, 0)
        self.figure_width_edit = QLineEdit()
        self.figure_width_edit.setValidator(self.int_validator)
        self.figure_width_edit.setText("800")
        self.main_layout.addWidget(self.figure_width_edit, 1, 1)
        self.figure_height_label = QLabel(self.tr("Height"))
        self.main_layout.addWidget(self.figure_height_label, 1, 2)
        self.figure_height_edit = QLineEdit()
        self.figure_height_edit.setValidator(self.int_validator)
        self.figure_height_edit.setText("600")
        self.main_layout.addWidget(self.figure_height_edit, 1, 3)
        self.pixel_ratio_label = QLabel(self.tr("Pixel Ratio"))
        self.main_layout.addWidget(self.pixel_ratio_label, 2, 0)
        self.pixel_ratio_edit = QLineEdit()
        self.pixel_ratio_edit.setValidator(self.double_validator)
        self.pixel_ratio_edit.setText("1.0")
        self.main_layout.addWidget(self.pixel_ratio_edit, 2, 1)
        self.format_options = {self.tr("Windows Bitmap (*.bmp)"): "BMP",
                               self.tr("Joint Photographic Experts Group (*.jpg)"): "JPG",
                               self.tr("Portable Network Graphics (*.png)"): "PNG",
                               self.tr("Portable Pixmap (*.ppm)"): "PPM",
                               self.tr("X11 Bitmap (*.xbm)"): "XBM",
                               self.tr("X11 Pixmap (*.xpm)"): "XPM",
                               self.tr("Scalable Vector Graphics (*.svg)"): "SVG"}
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.main_layout.addWidget(self.cancel_button, 3, 0, 1, 2)
        self.export_button = QPushButton(self.tr("Export"))
        self.main_layout.addWidget(self.export_button, 3, 2, 1, 2)

        self.figure_width_edit.textChanged.connect(self.update_preview)
        self.figure_height_edit.textChanged.connect(self.update_preview)
        self.pixel_ratio_edit.textChanged.connect(self.update_preview)

        self.cancel_button.clicked.connect(self.close)
        self.export_button.clicked.connect(self.save_figure)


    def show_message(self, title: str, message: str):
        self.msg_box.setWindowTitle(title)
        self.msg_box.setText(message)
        self.msg_box.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    def update_preview(self):
        try:
            width = int(self.figure_width_edit.text())
            height = int(self.figure_height_edit.text())
            pixel_ratio = float(self.pixel_ratio_edit.text())
        # the text may be empty and hence raise `ValueError`
        except ValueError:
            return
        # some invalid values may raise during typing process
        if width < 100 or height < 100 or pixel_ratio < 1.0:
            return
        pixmap = self.canvas.get_pixmap(width=width, height=height, pixel_ratio=pixel_ratio)
        preview_width = self.preview_label.width()
        preview_height = self.preview_label.height()
        preview_pixel_ratio = min(width*pixel_ratio/preview_width, height*pixel_ratio/preview_height)
        pixmap.setDevicePixelRatio(preview_pixel_ratio)
        self.preview_label.setPixmap(pixmap)

    def save_figure(self):
        width_str = self.figure_width_edit.text()
        height_str = self.figure_height_edit.text()
        pixel_ratio_str = self.pixel_ratio_edit.text()

        if width_str == "":
            self.show_warning(self.tr("The width should not be empty."))
            return
        if height_str == "":
            self.show_warning(self.tr("The height should not be empty."))
            return
        if pixel_ratio_str == "":
            self.show_warning(self.tr("The pixel ratio should not be empty."))
            return

        width = int(width_str)
        height = int(height_str)
        pixel_ratio = float(pixel_ratio_str)
        if width < 100:
            self.show_warning(self.tr("The width should be equal or greater than 100."))
            return
        if height < 100:
            self.show_warning(self.tr("The height should be equal or greater than 100."))
            return
        if pixel_ratio < 1.0:
            self.show_warning(self.tr("The pixel ratio should be equal or greater than 1.0."))
            return

        desktop_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
        filename, format_name = self.file_dialog.getSaveFileName(
            self, self.tr("Select Filename"),
            desktop_path, ";;".join([name for name, value in self.format_options.items()]))
        if self.format_options[format_name] in ("BMP", "JPG", "PNG", "PPM", "XBM", "XPM"):
            self.canvas.export_pixmap(filename, width=width, height=height, pixel_ratio=pixel_ratio)
        elif self.format_options[format_name] == "SVG":
            self.canvas.export_to_svg(filename)
        else:
            raise NotImplementedError(self.format_options[format_name])

