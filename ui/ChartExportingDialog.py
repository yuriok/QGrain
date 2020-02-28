import sys
from typing import Optional, Union

import numpy as np
from PySide2.QtCore import QSettings, QStandardPaths, Qt
from PySide2.QtGui import QDoubleValidator, QIntValidator
from PySide2.QtWidgets import (QDialog, QFileDialog, QGridLayout, QLabel,
                               QLineEdit, QPushButton)


class ChartExportingDialog(QDialog):
    def __init__(self, canvas, setting_group: str):
        super().__init__(parent=canvas)
        self.canvas = canvas
        self.settings = QSettings("./settings/chart_exporting.ini", QSettings.Format.IniFormat)
        self.settings.beginGroup(setting_group)
        self.file_dialog = QFileDialog(self)
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
        self.double_validator.setRange(1.00, 10.00, 2)
        self.width_label = QLabel(self.tr("Width"))
        self.main_layout.addWidget(self.width_label, 1, 0)
        self.width_edit = QLineEdit()
        self.width_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.width_edit, 1, 1)
        self.height_label = QLabel(self.tr("Height"))
        self.main_layout.addWidget(self.height_label, 1, 2)
        self.height_edit = QLineEdit()
        self.height_edit.setValidator(self.int_validator)
        self.main_layout.addWidget(self.height_edit, 1, 3)
        self.pixel_ratio_label = QLabel(self.tr("Pixel Ratio"))
        self.main_layout.addWidget(self.pixel_ratio_label, 2, 0)
        self.pixel_ratio_edit = QLineEdit()
        self.pixel_ratio_edit.setValidator(self.double_validator)
        self.main_layout.addWidget(self.pixel_ratio_edit, 2, 1)
        self.format_options = {self.tr("Windows Bitmap (*.bmp)"): "BMP",
                               self.tr("Joint Photographic Experts Group (*.jpg)"): "JPG",
                               self.tr("Portable Network Graphics (*.png)"): "PNG",
                               self.tr("Portable Pixmap (*.ppm)"): "PPM",
                               self.tr("X11 Bitmap (*.xbm)"): "XBM",
                               self.tr("X11 Pixmap (*.xpm)"): "XPM",
                               self.tr("Scalable Vector Graphics (*.svg)"): "SVG"}
        self.support_formats = [name for description, name in self.format_options.items()]
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.main_layout.addWidget(self.cancel_button, 3, 0, 1, 2)
        self.export_button = QPushButton(self.tr("Export"))
        self.main_layout.addWidget(self.export_button, 3, 2, 1, 2)

        self.width_edit.textChanged.connect(self.update)
        self.height_edit.textChanged.connect(self.update)
        self.pixel_ratio_edit.textChanged.connect(self.update)

        self.cancel_button.clicked.connect(self.close)
        self.export_button.clicked.connect(lambda: self.save_figure())

        self.restore()

    def restore(self):
        width = self.settings.value("width", defaultValue="800")
        height = self.settings.value("height", defaultValue="600")
        pixel_ratio = self.settings.value("pixel_ratio", defaultValue="1.0")
        # must end group before setting texts
        # because it will modify the settings
        self.width_edit.setText(width)
        self.height_edit.setText(height)
        self.pixel_ratio_edit.setText(pixel_ratio)

    @property
    def width(self) -> int:
        return self.settings.value("width", defaultValue=800, type=int)

    @width.setter
    def width(self, value: Union[int, str]):
        if isinstance(value, str) and value != "":
            value = int(value)
        if isinstance(value, int) and 100 <= value <= 10000:
            self.settings.setValue("width", value)

    @property
    def height(self) -> int:
        return self.settings.value("height", defaultValue=600, type=int)

    @height.setter
    def height(self, value: Union[int, str]):
        if isinstance(value, str) and value != "":
            value = int(value)
        if isinstance(value, int) and 100 <= value <= 10000:
            self.settings.setValue("height", value)

    @property
    def pixel_ratio(self) -> float:
        return self.settings.value("pixel_ratio", defaultValue=1.0, type=float)

    @pixel_ratio.setter
    def pixel_ratio(self, value: Union[int, float, str]):
        if isinstance(value, str) and value != "":
            value = float(value)
        if isinstance(value, (int, float)) and 1.0 <= value <= 100.0:
            self.settings.setValue("pixel_ratio", value)

    @property
    def last_format(self) -> str:
        return self.settings.value("last_format", defaultValue="png", type=str)

    @last_format.setter
    def last_format(self, value: str):
        if isinstance(value, str) and value in self.support_formats:
            self.settings.setValue("last_format", value)

    def update(self):
        # save to settings
        self.width = self.width_edit.text()
        self.height = self.height_edit.text()
        self.pixel_ratio = self.pixel_ratio_edit.text()

        pixmap = self.canvas.get_pixmap(width=self.width, height=self.height,
                                        pixel_ratio=self.pixel_ratio)
        preview_width = self.preview_label.width()
        preview_height = self.preview_label.height()
        preview_pixel_ratio = min(self.width*self.pixel_ratio/preview_width,
                                  self.height*self.pixel_ratio/preview_height)
        pixmap.setDevicePixelRatio(preview_pixel_ratio)
        self.preview_label.setPixmap(pixmap)

    def save_figure(self, filename: str = None):
        if filename is None:
            desktop_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
            filename, format_description = self.file_dialog.getSaveFileName(
                self, self.tr("Select Filename"),
                desktop_path, ";;".join([description for description, name in self.format_options.items()]))
            format_name = self.format_options[format_description]
        else:
            format_name = filename.split(".")[-1].upper()

        if format_name in ("BMP", "JPG", "PNG", "PPM", "XBM", "XPM"):
            self.canvas.export_pixmap(filename, width=self.width, height=self.height,
                                      pixel_ratio=self.pixel_ratio)
            self.last_format = format_name
        elif format_name == "SVG":
            self.canvas.export_to_svg(filename)
            self.last_format = format_name
        else:
            raise NotImplementedError(format_name)
