import csv
import logging
import os

import numpy as np
import pyqtgraph as pg
import xlsxwriter
import xlwt
from palettable.cartocolors.qualitative import Bold_10 as LightPalette
from palettable.cartocolors.qualitative import Pastel_10 as DarkPalette
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QFont, QRegExpValidator
from PySide2.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QWidget)
from sklearn.decomposition import PCA

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
from models.SampleDataset import SampleDataset


class PCAPanel(QWidget):
    logger = logging.getLogger("root.ui.PCAPanel")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, light=True, **kargs):
        super().__init__(parent, **kargs)
        if light:
            pg.setConfigOptions(foreground=pg.mkColor("k"))
        else:
            pg.setConfigOptions(foreground=pg.mkColor("w"))
        self.init_ui(light)

        self.file_dialog = QFileDialog(self)
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.retry_msg_box = QMessageBox(self)
        self.retry_msg_box.addButton(QMessageBox.StandardButton.Retry)
        self.retry_msg_box.addButton(QMessageBox.StandardButton.Ok)
        self.retry_msg_box.setDefaultButton(QMessageBox.StandardButton.Retry)
        self.retry_msg_box.setWindowFlags(Qt.Drawer)

        self.plot_data_items = []
        self.dataset = None
        self.transformed = None

    def init_ui(self, light: bool):
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget(enableMenu=False)
        self.main_layout.addWidget(self.plot_widget, 0, 0, 1, 5)
        # add image exporters
        self.png_exporter = ImageExporter(self.plot_widget.plotItem)
        self.svg_exporter = SVGExporter(self.plot_widget.plotItem)
        # show all axis
        self.plot_widget.plotItem.showAxis("left")
        self.plot_widget.plotItem.showAxis("right")
        self.plot_widget.plotItem.showAxis("top")
        self.plot_widget.plotItem.showAxis("bottom")
        # prepare the styles
        if light:
            self.component_styles = [dict(pen=pg.mkPen(hex_color, width=2)) for hex_color in LightPalette.hex_colors]
        else:
            self.component_styles = [dict(pen=pg.mkPen(hex_color, width=2)) for hex_color in DarkPalette.hex_colors]
        # set labels
        # bug of pyqtgraph, can not perform the foreground to labels
        if light:
            self.label_styles = {"font-family": "Times New Roman", "color": "black"}
        else:
            self.label_styles = {"font-family": "Times New Roman", "color": "white"}
        self.plot_widget.plotItem.setLabel("left", self.tr("Transformed"), **self.label_styles)
        self.plot_widget.plotItem.setLabel("bottom", self.tr("Sample Index"), **self.label_styles)
        # set title
        self.title_format = """<font face="Times New Roman">%s</font>"""
        self.plot_widget.plotItem.setTitle(self.title_format % self.tr("PCA Canvas"))
        # show grids
        self.plot_widget.plotItem.showGrid(True, True)
        # set the font of ticks
        self.tickFont = QFont("Arial")
        self.tickFont.setPointSize(8)
        self.plot_widget.plotItem.getAxis("left").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("right").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("top").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("bottom").tickFont = self.tickFont
        # set auto SI
        self.plot_widget.plotItem.getAxis("left").enableAutoSIPrefix(enable=False)
        self.plot_widget.plotItem.getAxis("right").enableAutoSIPrefix(enable=False)
        self.plot_widget.plotItem.getAxis("top").enableAutoSIPrefix(enable=False)
        self.plot_widget.plotItem.getAxis("bottom").enableAutoSIPrefix(enable=False)
        # set legend
        self.legend_format = """<font face="Times New Roman">%s</font>"""
        self.legend = pg.LegendItem(offset=(80, 50))
        self.legend.setParentItem(self.plot_widget.plotItem)

        self.component_number_validator = QRegExpValidator(r"^[1-9]\d*$") # >= 1
        self.fraction_validator = QRegExpValidator(r"^(0(\.\d{1,4})?|1(\.0{1,4})?)$") # 0.0000 - 1.0000
        self.assign_component_number_checkbox = QCheckBox(self.tr("Assign Component Number"))
        self.assign_component_number_checkbox.setCheckState(Qt.Unchecked)
        self.checkbox_state_label = QLabel(self.tr("Least Information Fraction:"))
        self.param_edit = QLineEdit()
        self.param_edit.setValidator(self.fraction_validator)
        self.param_edit.setText("0.9")
        self.perform_button = QPushButton(self.tr("Perform"))
        self.save_button = QPushButton(self.tr("Save"))
        self.main_layout.addWidget(self.assign_component_number_checkbox, 1, 0)
        self.main_layout.addWidget(self.checkbox_state_label, 1, 1)
        self.main_layout.addWidget(self.param_edit, 1, 2)
        self.main_layout.addWidget(self.perform_button, 1, 3)
        self.main_layout.addWidget(self.save_button, 1, 4)
        # connect
        self.assign_component_number_checkbox.stateChanged.connect(self.on_assign_checkbox_changed)
        self.perform_button.clicked.connect(self.on_perform_clicked)
        self.save_button.clicked.connect(self.on_save_clicked)

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

    def on_assign_checkbox_changed(self, state: Qt.CheckState):
        if state == Qt.Checked:
            self.checkbox_state_label.setText(self.tr("Component Number:"))
            self.param_edit.setValidator(self.component_number_validator)
            self.param_edit.setText("1")
        else:
            self.checkbox_state_label.setText(self.tr("Least Information Fraction:"))
            self.param_edit.setValidator(self.fraction_validator)
            self.param_edit.setText("0.9")

    def get_param_value(self):
        if self.param_edit.text() == "":
            self.show_warning(self.tr("The component number / least information fraction is necessary."))
            return None
        if self.assign_component_number_checkbox.checkState() == Qt.Checked:
            return int(self.param_edit.text())
        else:
            return float(self.param_edit.text())

    def on_data_loaded(self, dataset: SampleDataset):
        self.dataset = dataset

    def get_data_matrix(self):
        if self.dataset is None:
            self.show_warning(self.tr("Please load data first."))
            return None
        matrix = []
        for sample in self.dataset.samples:
            matrix.append(sample.distribution)
        return np.array(matrix)

    def on_perform_clicked(self):
        n_components = self.get_param_value()
        if n_components is None:
            return
        matrix = self.get_data_matrix()
        if matrix is None:
            return
        self.logger.debug("Start to perform PCA algorithm, n_components is [%s].", n_components)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(matrix)
        sample_number, dimension_number = transformed.shape
        x = np.array(range(1, sample_number+1))
        # clear
        for item in self.plot_data_items:
            self.plot_widget.plotItem.removeItem(item)
            self.legend.removeItem(item)
        self.plot_data_items.clear()
        for i in range(dimension_number):
            item_name = self.tr("Component") + " " + str(i+1)
            plot_data_item = pg.PlotDataItem(name=item_name)
            plot_data_item.setData(
                x, transformed[:, i],
                **self.component_styles[i % len(self.component_styles)])
            self.plot_widget.plotItem.addItem(plot_data_item)
            self.legend.addItem(plot_data_item, self.legend_format % item_name)
            self.plot_data_items.append(plot_data_item)

        self.transformed = transformed
        self.logger.debug("PCA algorithm performed.")

    def on_save_clicked(self):
        if self.transformed is None:
            self.show_warning(self.tr("The PCA algorithm has not been performed."))
            return
        filename, type_str = self.file_dialog.getSaveFileName(None, self.tr("Save Recorded Data"), None, "Excel (*.xlsx);;97-2003 Excel (*.xls);;CSV (*.csv)")

        if filename is None or filename == "":
            self.logger.info("The path is None or empty, ignored.")
            return
        if os.path.exists(filename):
            self.logger.warning("This file has existed and will be replaced. Filename: %s.", filename)
        self.logger.info("File path to save is [%s].", filename)
        if ".xlsx" in type_str:
            self.save_as_xlsx(filename)
        elif "97-2003" in type_str:
            self.save_as_xls(filename)
        elif ".csv" in type_str:
            self.save_as_csv(filename)
        else:
            raise NotImplementedError(type_str)
        self.logger.info("PCA result has been saved to [%s].", filename)
        self.show_info(self.tr("PCA result has been saved to:\n[%s].") % filename)

    def save_as_csv(self, filename: str):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            sample_number, dimension_number = self.transformed.shape
            headers = ["Sample Name"] + ["Component {0}".format(i+1) for i in range(dimension_number)]
            w.writerow(headers)
            for i in range(sample_number):
                w.writerow([self.dataset.samples[i].name] + list(self.transformed[i]))

    def save_as_xls(self, filename: str):
        sample_number, dimension_number = self.transformed.shape
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("PCA Result")
        sheet.write(0, 0, "Sample Name")
        for i in range(dimension_number):
            sheet.write(0, i+1, "Component {0}".format(i+1))
        for sample_index in range(sample_number):
            sheet.write(sample_index+1, 0, self.dataset.samples[sample_index].name)
            for dimension_index in range(dimension_number):
                sheet.write(sample_index+1, dimension_index+1, self.transformed[sample_index, dimension_index])
        workbook.save(filename)

    def save_as_xlsx(self, filename: str):
        sample_number, dimension_number = self.transformed.shape
        workbook = xlsxwriter.Workbook(filename)
        sheet = workbook.add_worksheet("PCA Result")
        sheet.write(0, 0, "Sample Name")
        for i in range(dimension_number):
            sheet.write(0, i+1, "Component {0}".format(i+1))
        for sample_index in range(sample_number):
            sheet.write(sample_index+1, 0, self.dataset.samples[sample_index].name)
            for dimension_index in range(dimension_number):
                sheet.write(sample_index+1, dimension_index+1, self.transformed[sample_index, dimension_index])
        workbook.close()

if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication
    pg.setConfigOptions(background=pg.mkColor("#ffffff00"))
    app = QApplication(sys.argv)
    panel = PCAPanel(light=True)
    panel.show()
    sys.exit(app.exec_())
