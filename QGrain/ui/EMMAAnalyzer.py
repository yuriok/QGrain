__all__ = ["EMMAAnalyzer"]

import logging
import os
import pickle
import typing

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..chart.EMMAResultChart import EMMAResultChart
from ..emma import EMMAResolver, EMMAResult, KernelType
from ..io import save_emma
from ..model import GrainSizeDataset
from .EMMASettingDialog import EMMASettingDialog
from .ParameterEditor import ParameterEditor


class EMMAAnalyzer(QtWidgets.QWidget):
    logger = logging.getLogger("QGrain")
    SUPPORT_KERNELS = (
        KernelType.Nonparametric,
        KernelType.Normal,
        KernelType.SkewNormal,
        KernelType.Weibull,
        KernelType.GeneralWeibull)
    def __init__(self, setting_dialog: EMMASettingDialog,
                 parameter_editor: ParameterEditor, parent=None):
        super().__init__(parent=parent)
        assert setting_dialog is not None
        assert parameter_editor is not None
        self.setting_dialog = setting_dialog
        self.parameter_editor = parameter_editor
        self.init_ui()
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.__dataset = None # type: GrainSizeDataset
        self.__result_list = [] # type: list[EMMAResult]
        self.file_dialog = QtWidgets.QFileDialog(parent=self)

    def init_ui(self):
        self.setWindowTitle(self.tr("EMMA Analyzer"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        # Control group
        self.control_group = QtWidgets.QGroupBox(self.tr("Control"))
        self.control_layout = QtWidgets.QGridLayout(self.control_group)
        self.kernel_label = QtWidgets.QLabel(self.tr("Kernel Type"))
        self.kernel_combo_box = QtWidgets.QComboBox()
        self.kernel_combo_box.addItems([kernel_type.value for kernel_type in self.SUPPORT_KERNELS])
        self.kernel_combo_box.setCurrentIndex(1)
        self.control_layout.addWidget(self.kernel_label, 0, 0)
        self.control_layout.addWidget(self.kernel_combo_box, 0, 1)
        self.n_members_label = QtWidgets.QLabel("Number of End Members")
        self.n_members_input = QtWidgets.QSpinBox()
        self.n_members_input.setRange(1, 12)
        self.n_members_input.setValue(3)
        self.control_layout.addWidget(self.n_members_label, 1, 0)
        self.control_layout.addWidget(self.n_members_input, 1, 1)
        self.update_EMs_checkbox = QtWidgets.QCheckBox(self.tr("Update Distributions of End Members"))
        self.update_EMs_checkbox.setChecked(True)
        self.control_layout.addWidget(self.update_EMs_checkbox, 2, 0, 1, 2)
        self.try_fit_button = QtWidgets.QPushButton(self.tr("Try Fit"))
        self.try_fit_button.clicked.connect(self.on_try_fit_clicked)
        self.try_fit_button.setEnabled(False)
        self.edit_parameter_button = QtWidgets.QPushButton(self.tr("Edit Parameters"))
        self.edit_parameter_button.clicked.connect(self.on_edit_parameter_clicked)
        self.edit_parameter_button.setEnabled(False)
        self.control_layout.addWidget(self.try_fit_button, 3, 0)
        self.control_layout.addWidget(self.edit_parameter_button, 3, 1)

        # Result group
        self.result_group = QtWidgets.QGroupBox(self.tr("Result"))
        self.result_layout = QtWidgets.QGridLayout(self.result_group)
        self.result_list_widget = QtWidgets.QListWidget()
        self.result_layout.addWidget(self.result_list_widget, 0, 0, 1, 2)
        self.remove_result_button = QtWidgets.QPushButton(self.tr("Remove"))
        self.remove_result_button.clicked.connect(self.on_remove_clicked)
        self.try_summarize_button = QtWidgets.QPushButton(self.tr("Try Summarize"))
        self.try_summarize_button.clicked.connect(self.on_try_summarize_clicked)
        self.show_result_button = QtWidgets.QPushButton(self.tr("Show Result"))
        self.show_result_button.clicked.connect(self.on_show_clicked)
        self.show_animation_button = QtWidgets.QPushButton(self.tr("Show Animation"))
        self.show_animation_button.clicked.connect(self.on_show_animation_clicked)
        self.load_button = QtWidgets.QPushButton(self.tr("Load"))
        self.load_button.clicked.connect(self.on_load_clicked)
        self.save_button = QtWidgets.QPushButton(self.tr("Save"))
        self.save_button.clicked.connect(self.on_save_clicked)
        self.remove_result_button.setEnabled(False)
        self.try_summarize_button.setEnabled(False)
        self.show_result_button.setEnabled(False)
        self.show_animation_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.result_layout.addWidget(self.remove_result_button, 1, 0)
        self.result_layout.addWidget(self.try_summarize_button, 1, 1)
        self.result_layout.addWidget(self.show_result_button, 2, 0)
        self.result_layout.addWidget(self.show_animation_button, 2, 1)
        self.result_layout.addWidget(self.load_button, 3, 0)
        self.result_layout.addWidget(self.save_button, 3, 1)
        self.chart_group = QtWidgets.QGroupBox(self.tr("Chart"))
        self.chart_layout = QtWidgets.QGridLayout(self.chart_group)
        self.result_chart = EMMAResultChart()
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
        resolver = EMMAResolver()
        resolver_setting = self.setting_dialog.setting
        update_end_members = self.update_EMs_checkbox.isChecked()
        if self.parameter_editor.parameter_enabled:
            kernel_type = KernelType.__members__[self.parameter_editor.distribution_type.name]
            parameters = self.parameter_editor.parameters[:-1, :].astype(np.float32)
            result = resolver.try_fit(self.__dataset, kernel_type, self.parameter_editor.n_components, resolver_setting, parameters, update_end_members)
        else:
            result = resolver.try_fit(self.__dataset, self.kernel_type, self.n_members, resolver_setting, update_end_members=update_end_members)
        self.add_results([result])
        self.result_chart.show_result(result)
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)

    def get_result_name(self, result: EMMAResult):
        if self.update_EMs_checkbox.isChecked():
            fixed_str = " "
        else:
            fixed_str = " Fixed "
        return f"{result.n_members}{fixed_str}{result.kernel_type.value}"

    def add_results(self, results: typing.List[EMMAResult]):
        if self.n_results == 0:
            self.remove_result_button.setEnabled(True)
            self.try_summarize_button.setEnabled(True)
            self.show_result_button.setEnabled(True)
            self.show_animation_button.setEnabled(True)
            self.save_button.setEnabled(True)

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
            self.remove_result_button.setEnabled(False)
            self.try_summarize_button.setEnabled(False)
            self.show_result_button.setEnabled(False)
            self.show_animation_button.setEnabled(False)
            self.save_button.setEnabled(False)

    def on_try_summarize_clicked(self):
        pass

    def on_show_clicked(self):
        result = self.selected_result
        if result is not None:
            self.result_chart.show_result(result)

    def on_show_animation_clicked(self):
        result = self.selected_result
        if result is not None:
            self.result_chart.show_animation(result)

    def on_load_clicked(self):
        filename, _  = self.file_dialog.getOpenFileName(
            self, self.tr("Select the dump file of the EMMA result(s)"),
            None, f"{self.tr('Binary Dump')} (*.dump)")
        if filename is None or filename == "":
            return
        with open(filename, "rb") as f:
            results = pickle.load(f)
            invalid = False
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, EMMAResult):
                        invalid = True
                        break
            else:
                invalid = True
            if invalid:
                self.show_error(self.tr("The dump file does not contain any EMMA result."))
                return
            else:
                self.add_results(results)

    def on_save_clicked(self):
        if self.n_results == 0:
            self.show_warning(self.tr("There is not an EMMA result in the list."))
            return

        filename, _ = self.file_dialog.getSaveFileName(
            self, self.tr("Choose a filename to save the EMMA result(s) in list"),
            None, f"{self.tr('Binary Dump')} (*.dump);;{self.tr('Microsoft Excel')} (*.xlsx)")
        if filename is None or filename == "":
            return
        _, ext = os.path.splitext(filename)

        if ext == ".dump":
            with open(filename, "wb") as f:
                pickle.dump(self.__result_list, f)
                self.logger.info("All EMMA results in list has been saved to the dump file.")
        elif ext == ".xlsx":
            try:
                result = self.selected_result
                progress_dialog = QtWidgets.QProgressDialog(
                    self.tr("Saving EMMA result..."), self.tr("Cancel"),
                    0, 100, self)
                progress_dialog.setWindowTitle("QGrain")
                progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
                def callback(progress: float):
                    if progress_dialog.wasCanceled():
                        raise StopIteration()
                    progress_dialog.setValue(int(progress*100))
                    QtCore.QCoreApplication.processEvents()
                save_emma(result, filename, progress_callback=callback)
                progress_dialog.setValue(100)
                self.logger.info("The selected EMMA result of this dataset has been saved to the Excel file.")
            except Exception as e:
                self.logger.exception("An unknown exception was raised. Please check the logs for more details.", stack_info=True)
                self.show_error(self.tr("An unknown exception was raised. Please check the logs for more details."))

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("EMMA Analyzer"))
        self.control_group.setTitle(self.tr("Control"))
        self.kernel_label.setText(self.tr("Kernel Type"))
        self.n_members_label.setText("Number of End Members")
        self.update_EMs_checkbox.setText(self.tr("Update Distributions of End Members"))
        self.try_fit_button.setText(self.tr("Try Fit"))
        self.edit_parameter_button.setText(self.tr("Edit Parameters"))
        self.result_group.setTitle(self.tr("Result"))
        self.remove_result_button.setText(self.tr("Remove"))
        self.try_summarize_button.setText(self.tr("Try Summarize"))
        self.show_result_button.setText(self.tr("Show Result"))
        self.show_animation_button.setText(self.tr("Show Animation"))
        self.load_button.setText(self.tr("Load"))
        self.save_button.setText(self.tr("Save"))
        self.chart_group.setTitle(self.tr("Chart"))
