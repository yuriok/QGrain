__all__ = ["UDMAnalyzer"]

import logging
import os
import pickle
import typing

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..chart.UDMResultChart import UDMResultChart
from ..io import save_udm
from ..model import GrainSizeDataset
from ..udm import KernelType, UDMResolver, UDMResult
from .ParameterEditor import ParameterEditor
from .UDMSettingDialog import UDMSettingDialog


class UDMAnalyzer(QtWidgets.QWidget):
    logger = logging.getLogger("QGrain.UDMAnalyzer")
    SUPPORT_KERNELS = (
        KernelType.Normal,
        KernelType.SkewNormal,
        KernelType.Weibull,
        KernelType.GeneralWeibull)
    def __init__(self, setting_dialog: UDMSettingDialog,
                 parameter_editor: ParameterEditor, parent=None):
        super().__init__(parent=parent)
        assert setting_dialog is not None
        assert parameter_editor is not None
        self.setting_dialog = setting_dialog
        self.parameter_editor = parameter_editor
        self.init_ui()
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.__dataset = None # type: GrainSizeDataset
        self.__result_list = [] # type: list[UDMResult]
        self.file_dialog = QtWidgets.QFileDialog(parent=self)

    def init_ui(self):
        self.setWindowTitle(self.tr("UDM Analyzer"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        # Control group
        self.control_group = QtWidgets.QGroupBox(self.tr("Control"))
        self.control_layout = QtWidgets.QGridLayout(self.control_group)
        self.kernel_label = QtWidgets.QLabel(self.tr("Kernel Type"))
        self.kernel_combo_box = QtWidgets.QComboBox()
        self.kernel_combo_box.addItems([kernel_type.value for kernel_type in self.SUPPORT_KERNELS])
        self.kernel_combo_box.setCurrentIndex(0)
        self.control_layout.addWidget(self.kernel_label, 0, 0)
        self.control_layout.addWidget(self.kernel_combo_box, 0, 1)
        self.n_components_label = QtWidgets.QLabel(self.tr("Number of Components"))
        self.n_members_input = QtWidgets.QSpinBox()
        self.n_members_input.setRange(1, 12)
        self.n_members_input.setValue(3)
        self.control_layout.addWidget(self.n_components_label, 1, 0)
        self.control_layout.addWidget(self.n_members_input, 1, 1)

        self.try_fit_button = QtWidgets.QPushButton(self.tr("Try Fit"))
        self.try_fit_button.clicked.connect(self.on_try_fit_clicked)
        self.try_fit_button.setEnabled(False)
        self.edit_parameter_button = QtWidgets.QPushButton(self.tr("Edit Parameters"))
        self.edit_parameter_button.clicked.connect(self.on_edit_parameter_clicked)
        self.edit_parameter_button.setEnabled(False)
        self.control_layout.addWidget(self.try_fit_button, 2, 0)
        self.control_layout.addWidget(self.edit_parameter_button, 2, 1)

        # Result group
        self.result_group = QtWidgets.QGroupBox(self.tr("Result"))
        self.result_layout = QtWidgets.QGridLayout(self.result_group)
        self.result_list_widget = QtWidgets.QListWidget()
        self.result_layout.addWidget(self.result_list_widget, 0, 0, 1, 2)
        self.remove_button = QtWidgets.QPushButton(self.tr("Remove"))
        self.remove_button.clicked.connect(self.on_remove_clicked)
        self.show_button = QtWidgets.QPushButton(self.tr("Show"))
        self.show_button.clicked.connect(self.on_show_clicked)
        self.remove_button.setEnabled(False)
        self.show_button.setEnabled(False)
        self.result_layout.addWidget(self.remove_button, 1, 0)
        self.result_layout.addWidget(self.show_button, 1, 1)
        self.chart_group = QtWidgets.QGroupBox(self.tr("Chart"))
        self.chart_layout = QtWidgets.QGridLayout(self.chart_group)
        self.result_chart = UDMResultChart()
        self.chart_layout.addWidget(self.result_chart)

        self.splitter_1 = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter_1.addWidget(self.control_group)
        self.splitter_1.addWidget(self.result_group)
        self.splitter_2 = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter_2.addWidget(self.splitter_1)
        self.splitter_2.addWidget(self.chart_group)
        self.main_layout.addWidget(self.splitter_2)

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

    @property
    def kernel_type(self) -> KernelType:
        kernel_type = self.SUPPORT_KERNELS[self.kernel_combo_box.currentIndex()]
        return kernel_type

    @property
    def n_members(self):
        return self.n_members_input.value()

    @property
    def n_results(self) -> int:
        return len(self.__result_list)

    @property
    def selected_index(self):
        indexes = self.result_list_widget.selectedIndexes()
        if len(indexes) == 0:
            return 0
        else:
            return indexes[0].row()

    @property
    def selected_result(self):
        if self.n_results > 0:
            return self.__result_list[self.selected_index]

    def on_dataset_loaded(self, dataset: GrainSizeDataset):
        self.__dataset = dataset
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)

    def on_edit_parameter_clicked(self):
        if self.__dataset is not None:
            target = np.mean(self.__dataset.distribution_matrix, axis=0)
            target = target / np.sum(target)
            self.parameter_editor.setup_target(self.__dataset.classes_μm, target)
        self.parameter_editor.show()

    def on_try_fit_clicked(self):
        if self.__dataset is None:
            self.show_error(self.tr("Dataset has not been loaded."))
            return
        self.try_fit_button.setEnabled(False)
        self.edit_parameter_button.setEnabled(False)
        resolver = UDMResolver()
        resolver_setting = self.setting_dialog.setting
        if self.parameter_editor.parameter_enabled:
            kernel_type = KernelType.__members__[self.parameter_editor.distribution_type.name]
            parameters = self.parameter_editor.parameters[:-1, :].astype(np.float32)
            result = resolver.try_fit(
                self.__dataset, kernel_type,
                self.parameter_editor.n_components,
                resolver_setting=resolver_setting,
                parameters=parameters)
        else:
            result = resolver.try_fit(
                self.__dataset, self.kernel_type,
                self.n_members,
                resolver_setting=resolver_setting)
        self.add_results([result])
        self.result_chart.show_result(result)
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)

    def get_result_name(self, result: UDMResult):
        return f"{result.n_components} {result.kernel_type.value}"

    def add_results(self, results: typing.List[UDMResult]):
        if self.n_results == 0:
            self.remove_button.setEnabled(True)
            self.show_button.setEnabled(True)

        self.__result_list.extend(results)
        self.result_list_widget.addItems([self.get_result_name(result) for result in results])

    def on_remove_clicked(self):
        if self.n_results == 0:
            return
        else:
            index = self.selected_index
            self.__result_list.pop(index)
            self.result_list_widget.takeItem(index)

        if self.n_results == 0:
            self.remove_button.setEnabled(False)
            self.show_button.setEnabled(False)

    def on_show_clicked(self):
        result = self.selected_result
        if result is not None:
            self.result_chart.show_result(result)

    def on_show_animation_clicked(self):
        result = self.selected_result
        if result is not None:
            self.result_chart.show_animation(result)

    def load_result(self):
        filename, _  = self.file_dialog.getOpenFileName(
            self, self.tr("Choose the file which stores the dumped UDM result"),
            None, "Dumped UDM Result (*.udm)")
        if filename is None or filename == "":
            return
        with open(filename, "rb") as f:
            result = pickle.load(f)
            valid = True
            if isinstance(result, UDMResult):
                self.add_results([result])
                self.logger.info(f"The dumped UDM result has been loaded.")
            else:
                self.logger.error("The binary file is invalid (i.e., the object in it is not the UDM result).")
                self.show_error(self.tr("The binary file is invalid (i.e., the object in it is not the UDM result)."))
                return

    def save_selected_result(self):
        if self.n_results == 0:
            self.show_warning(self.tr("There is no UDM result."))
            return
        filename, _ = self.file_dialog.getSaveFileName(
            self, self.tr("Choose a filename to save the selected UDM result"),
            None, "Microsoft Excel (*.xlsx);;Dumped UDM Result (*.udm)")
        if filename is None or filename == "":
            return
        try:
            # Excel
            if filename[-4:] == "xlsx":
                result = self.selected_result
                progress_dialog = QtWidgets.QProgressDialog(
                    self.tr("Saving the UDM result..."), self.tr("Cancel"),
                    0, 100, self)
                progress_dialog.setWindowTitle("QGrain")
                progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
                def callback(progress: float):
                    if progress_dialog.wasCanceled():
                        raise StopIteration()
                    progress_dialog.setValue(int(progress*100))
                    QtCore.QCoreApplication.processEvents()
                save_udm(result, filename, progress_callback=callback, logger=self.logger)
            # Binary File
            else:
                with open(filename, "wb") as f:
                    pickle.dump(self.selected_result, f)
                    self.logger.info("The selected UDM result has been dumped.")
        except Exception as e:
            self.logger.exception("An unknown exception was raised. Please check the logs for more details.", stack_info=True)
            self.show_error(self.tr("An unknown exception was raised. Please check the logs for more details."))

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("UDM Analyzer"))
        self.control_group.setTitle(self.tr("Control"))
        self.kernel_label.setText(self.tr("Kernel Type"))
        self.n_components_label.setText(self.tr("Number of Components"))
        self.try_fit_button.setText(self.tr("Try Fit"))
        self.edit_parameter_button.setText(self.tr("Edit Parameters"))
        self.result_group.setTitle(self.tr("Result"))
        self.remove_button.setText(self.tr("Remove"))
        self.show_button.setText(self.tr("Show"))
        self.chart_group.setTitle(self.tr("Chart"))
