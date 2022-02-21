__all__ = ["ReferenceAssembler"]

import logging
import pickle
import typing
from uuid import UUID

import numpy as np
import openpyxl

from PySide6.QtCore import QCoreApplication, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDialog, QFileDialog, QGridLayout, QGroupBox,
                               QLabel, QListWidget, QMenu, QMessageBox,
                               QPushButton, QSizePolicy, QTableWidget,
                               QTableWidgetItem)

from ..ssu import SSUResult, Reference


class ReferenceAssembler(QDialog):
    logger = logging.getLogger("QGrain.ui.ReferenceAssembler")

    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("Reference Assembler"))
        self.init_ui()
        self.results = [] # type: list[SSUResult]
        self.current_result = None
        self.selected_components = [] # type: list[tuple[SSUResult, int]]
        self.file_dialog = QFileDialog(self)

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.ssu_result_group = QGroupBox(self.tr("SSU Results"))
        self.ssu_result_layout = QGridLayout(self.ssu_result_group)
        self.ssu_result_list = QListWidget()
        self.ssu_result_load_button = QPushButton(self.tr("Load"))
        self.ssu_result_select_button = QPushButton(self.tr("Select"))
        self.ssu_result_layout.addWidget(self.ssu_result_list, 0, 0)
        self.ssu_result_layout.addWidget(self.ssu_result_load_button, 1, 0)
        self.ssu_result_layout.addWidget(self.ssu_result_select_button, 2, 0)
        self.main_layout.addWidget(self.ssu_result_group, 0, 0)

        self.component_group = QGroupBox(self.tr("Result Components"))
        self.component_layout = QGridLayout(self.component_group)
        self.component_list = QListWidget()
        self.component_select_button = QPushButton(self.tr("Select"))
        self.component_layout.addWidget(self.component_list, 0, 0)
        self.component_layout.addWidget(self.component_select_button, 1, 0)
        self.main_layout.addWidget(self.component_group, 0, 1)

        self.reference_group = QGroupBox(self.tr("Reference Components"))
        self.reference_layout = QGridLayout(self.reference_group)
        self.reference_list = QListWidget()
        self.reference_remove_button = QPushButton(self.tr("Remove"))
        self.reference_save_button = QPushButton(self.tr("Save"))
        self.reference_layout.addWidget(self.reference_list, 0, 0)
        self.reference_layout.addWidget(self.reference_remove_button, 1, 0)
        self.reference_layout.addWidget(self.reference_save_button, 2, 0)
        self.main_layout.addWidget(self.reference_group, 0, 2)

        self.ssu_result_load_button.clicked.connect(self.load_results)
        self.ssu_result_select_button.clicked.connect(self.select_ssu_result)
        self.component_select_button.clicked.connect(self.select_component)
        self.reference_remove_button.clicked.connect(self.remove_component)
        self.reference_save_button.clicked.connect(self.save_reference)

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

    def load_results(self):
        filename, _ = self.file_dialog.getOpenFileName(\
            self, self.tr("Select a file"), None,
            self.tr("SSU Result Dump File (*.dump)"))
        if filename is None or filename == "":
            return
        try:
            with open(filename, "rb") as f:
                ssu_results = pickle.load(f) # type: list[SSUResult]
            for result in ssu_results:
                if not isinstance(result, SSUResult):
                    self.show_error(self.tr("The objects in this dump file are not SSU results."))
                    return
            self.results.extend(ssu_results)
            for result in ssu_results:
                self.ssu_result_list.addItem(result.sample.name)
        except Exception as e:
            self.show_error(self.tr("There is a error raised while loading the SSU results in this dump file:\n    {0}") % e.__str__())
            return

    def select_ssu_result(self):
        index = self.ssu_result_list.currentRow()
        if index < 0:
            return
        result = self.results[index]
        self.current_result = result
        self.component_list.clear()
        for i, component in enumerate(result.components):
            self.component_list.addItem(f"C{i+1}")

    def select_component(self):
        if self.current_result is None:
            return
        index = self.component_list.currentRow()
        if index < 0:
            return
        if len(self.selected_components) > 0:
            if self.current_result.distribution_type != self.selected_components[0][0].distribution_type:
                self.show_error("The distribution type of current component is not same as that of selected components.")
                return
        self.selected_components.append((self.current_result, index))
        self.reference_list.addItem(f"{self.current_result.sample.name} - C{index+1}")

    def remove_component(self):
        index = self.reference_list.currentRow()
        if index < 0:
            return
        self.selected_components.pop(index)
        self.reference_list.takeItem(index)

    def save_reference(self):
        if len(self.selected_components) == 0:
            return
        filename, _ = self.file_dialog.getSaveFileName(\
            self, self.tr("Select a file"), None,
            self.tr("Reference Dump File (*.dump)"))
        if filename is None or filename == "":
            return

        combined_args = [np.expand_dims(result.components[index].component_args, axis=1) for result, index in self.selected_components]
        combined_args = np.concatenate(combined_args, axis=1)
        distribution_type = self.selected_components[0][0].distribution_type
        print(distribution_type, combined_args)

        reference = Reference(distribution_type, combined_args)
        with open(filename, "wb") as f:
            pickle.dump(reference, f)
