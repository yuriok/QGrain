import logging
import os.path
import sys

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QMutex
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter

from xlrd import open_workbook
from xlrd.biffh import XLRDError
from xlrd.book import Book
from xlrd.sheet import Sheet




class WorkbookLoader(QObject):
    sigWorkFinished = pyqtSignal(Book)

    def __init__(self):
        super().__init__()
        self.mutex = QMutex()

    def load_workbook(self, filename):
        if filename is None or filename == "":
            raise ValueError(filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        try:
            workbook = open_workbook(filename)
            self.sigWorkFinished.emit(workbook)
        except XLRDError:
            logging.exception(self.tr("The format is incorrect."))
            self.sigWorkFinished.emit(None)
        except Exception:
            logging.error(self.tr("An unknown exception occurred."), stack_info=True)
            self.sigWorkFinished.emit(None)


# The settings are to complex, abandon this implementation
class DataLoadParameter(GroupParameter):
    sigDataLoaded = pyqtSignal(np.ndarray, list)
    def __init__(self, **opts):
        opts["type"] = "bool"
        opts["value"] = False
        opts["name"] = "Detailed Parameters to Load Data"
        super().__init__(**opts)
        
        self.select_file_param = self.addChild({"name": self.tr("Select File"), "type": "action"})
        self.sheet_param = self.addChild({"name": self.tr("Sheet Name"), "type": "list", "values": {}, "default": 0})
        self.sample_id_column_param = self.addChild({"name": self.tr("Sample ID Column Index"), "type": "int", "default": 0, "value": 0})
        self.classes_row_param = self.addChild({"name": self.tr("Classes Row Index"), "type": "int", "default": 0, "value": 0})
        self.data_start_row_param = self.addChild(self.addChild({"name": self.tr("Data Start Row Index"), "type": "int", "default": 1, "value": 1}))
        self.data_start_column_param = self.addChild({"name": self.tr("Data Start Column Index"), "type": "int", "default": 1, "value": 1})
        self.preview_data_param = self.addChild({"name": self.tr("Preview Data"), "type": "action"})

        self.file_dialog = QFileDialog()
        self.workbook_loader = WorkbookLoader()
        self.workbook = None
        self.sheet = None
        self.workbook_loader.sigWorkFinished.connect(self.on_loading_work_finished)

        self.select_file_param.sigActivated.connect(self.on_select_file_clicked)
        self.preview_data_param.sigActivated.connect(self.on_preview_data_clicked)


    def on_select_file_clicked(self):
        filename, _ = self.file_dialog.getOpenFileName(None, self.tr("Select Excel File"), None, "*.xls; *.xlsx")
        logging.info(self.tr("File path is [{0}].").format(filename))
        if filename is None or filename == "":
            return
        if not os.path.exists(filename):
            return

        self.workbook_loader.load_workbook(filename)


    def on_loading_work_finished(self, workbook):
        if workbook is None:
            logging.error(self.tr("Workbook loading is failed."))
            return
        
        self.workbook = workbook
        self.update_sheet_param()


    def update_sheet_param(self):
        if self.workbook is None:
            logging.error(self.tr("The workbook is unexpected null."))
            return
        sheet_selections = {name: index for name, index in zip(self.workbook.sheet_names(), range(self.workbook.nsheets))}
        self.sheet_param.setName("DISCARDED SOON")
        newSheetParam = self.insertChild(self.sheet_param, {"name": self.tr("Sheet Name"), "type": "list", "values": sheet_selections, "default": 0})
        self.removeChild(self.sheet_param)
        self.sheet_param = newSheetParam

        self.sheet_param.sigValueChanged.connect(self.on_sheet_selected)
        # Initiatively call this method once, because the value will be set automatically.
        self.on_sheet_selected()


    def on_sheet_selected(self):
        logging.info(self.tr("Sheet index is {0}.").format(self.sheet_param.value()))
        self.sheet = self.workbook.sheet_by_index(self.sheet_param.value())


    def on_preview_data_clicked(self):
        data_index = 0
        # get the params from settings
        sample_id_column = self.sample_id_column_param.value()
        classes_row = self.classes_row_param.value()
        data_start_row = self.data_start_row_param.value()
        data_start_column = self.data_start_column_param.value()

        # To get the x series
        grain_size_classes = np.array(self.sheet.row_values(classes_row)[data_start_column:])
        data = []
        for i in range(data_start_row, self.sheet.nrows):
            row_values = self.sheet.row_values(i)
            sample_id = row_values[sample_id_column]
            sample_data = np.array(row_values[data_start_column:])
            data.append({"id": sample_id, "data": sample_data})

        self.sigDataLoaded.emit(grain_size_classes, data)
        
