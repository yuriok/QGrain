import csv
import logging
import os

import numpy as np
import xlsxwriter
import xlwt
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import Qt
from PySide2.QtGui import QRegExpValidator
from PySide2.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QWidget)
from sklearn.decomposition import PCA


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())
from models.SampleDataset import SampleDataset
from ui.Canvas import Canvas


class PCAPanel(Canvas):
    logger = logging.getLogger("root.ui.PCAPanel")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, isDark=True):
        super().__init__(parent)
        self.initChart()
        self.setThemeMode(isDark)
        self.setupChartStyle()

        self.chart.legend().detachFromChart()
        self.chart.legend().setPos(100.0, 60.0)

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

    def initUI(self):
        super().initUI()
        # use anotehr widget to pack other controls
        self.controlContainer = QWidget(self)
        self.controlLayout = QGridLayout(self.controlContainer)
        self.controlLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addWidget(self.controlContainer, 1, 0)
        # prepare controls
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
        self.controlLayout.addWidget(self.assign_component_number_checkbox, 0, 0)
        self.controlLayout.addWidget(self.checkbox_state_label, 0, 1)
        self.controlLayout.addWidget(self.param_edit, 0, 2)
        self.controlLayout.addWidget(self.perform_button, 0, 3)
        self.controlLayout.addWidget(self.save_button, 0, 4)
        # connect
        self.assign_component_number_checkbox.stateChanged.connect(self.on_assign_checkbox_changed)
        self.perform_button.clicked.connect(self.on_perform_clicked)
        self.save_button.clicked.connect(self.on_save_clicked)

    def initChart(self):
        # init axes
        self.axisX = QtCharts.QValueAxis()
        self.axisX.setLabelFormat("%i")
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.axisY = QtCharts.QValueAxis()
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        # set title
        self.chart.setTitle(self.tr("PCA Canvas"))
        # set labels
        self.axisX.setTitleText(self.tr("Sample Index"))
        self.axisY.setTitleText(self.tr("Transformed"))
        # use demo to let it perform normal
        self.showDemo(self.axisX, self.axisY)

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
        self.on_perform_clicked()

    def get_data_matrix(self):
        if self.dataset is None:
            self.show_warning(self.tr("Please load data first."))
            return None
        matrix = []
        for sample in self.dataset.samples:
            matrix.append(sample.distribution)
        return np.array(matrix)

    def on_perform_clicked(self):
        # necessary to stop
        self.stopDemo()
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
        self.chart.removeAllSeries()
        for i in range(dimension_number):
            componentName = self.tr("Component") + " " + str(i+1)
            series = QtCharts.QLineSeries()
            series.setName(componentName)
            series.replace(self.toPoints(x, transformed[:, i]))
            self.chart.addSeries(series)
            series.attachAxis(self.axisX)
            series.attachAxis(self.axisY)
        # update the size of legend
        self.chart.legend().setMinimumSize(150.0, 30*(2+dimension_number))
        # reset the range of axes
        self.axisX.setRange(x[0], x[-1])
        self.axisY.setRange(np.min(transformed), np.max(transformed))

        self.transformed = transformed
        self.logger.debug("PCA algorithm performed.")
        # export chart to file
        self.exportToPng("./temp/pca_panel.png")
        self.exportToSvg("./temp/pca_panel.svg")

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
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    panel = PCAPanel(isDark=False)
    panel.chart.legend().hide()
    panel.show()
    sys.exit(app.exec_())
