__all__ = ["EMMAAnalyzer"]

import logging
import pickle
from typing import *

import numpy as np
from PySide6 import QtCore, QtWidgets
from grpc._channel import _InactiveRpcError

from ..models import KernelType, Dataset, EMMAResult
from ..charts.EMMAResultChart import EMMAResultChart
from ..protos.client import QGrainClient
from ..io import save_emma
from .EMMASettings import EMMASettings
from .ParameterEditor import ParameterEditor


class EMMAAnalyzer(QtWidgets.QWidget):
    logger = logging.getLogger("QGrain.EMMAAnalyzer")
    SUPPORT_KERNELS = (
        (KernelType.Nonparametric, "Nonparametric"),
        (KernelType.Normal, "Normal"),
        (KernelType.SkewNormal, "Skew Normal"),
        (KernelType.Weibull, "Weibull"),
        (KernelType.GeneralWeibull, "General Weibull"))

    def __init__(self, setting_dialog: EMMASettings, parameter_editor: ParameterEditor,
                 client: QGrainClient = None, parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)
        assert setting_dialog is not None
        assert parameter_editor is not None
        self.setting_dialog = setting_dialog
        self.parameter_editor = parameter_editor
        self._client = client
        self.setWindowTitle(self.tr("EMMA Analyzer"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.control_group = QtWidgets.QGroupBox(self.tr("Control"))
        self.control_group.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.control_layout = QtWidgets.QGridLayout(self.control_group)
        self.kernel_label = QtWidgets.QLabel(self.tr("Kernel Type"))
        self.kernel_combo_box = QtWidgets.QComboBox()
        self.kernel_combo_box.addItems([name for _, name in self.SUPPORT_KERNELS])
        self.kernel_combo_box.setCurrentIndex(1)
        self.control_layout.addWidget(self.kernel_label, 0, 0)
        self.control_layout.addWidget(self.kernel_combo_box, 0, 1)
        self.n_members_label = QtWidgets.QLabel(self.tr("Number of End Members"))
        self.n_members_input = QtWidgets.QSpinBox()
        self.n_members_input.setRange(1, 12)
        self.n_members_input.setValue(3)
        self.control_layout.addWidget(self.n_members_label, 1, 0)
        self.control_layout.addWidget(self.n_members_input, 1, 1)
        self.try_fit_button = QtWidgets.QPushButton(self.tr("Try Fit"))
        self.try_fit_button.clicked.connect(self.on_try_fit_clicked)
        self.try_fit_button.setEnabled(False)
        self.edit_parameter_button = QtWidgets.QPushButton(self.tr("Edit Parameters"))
        self.edit_parameter_button.clicked.connect(self.on_edit_parameter_clicked)
        self.edit_parameter_button.setEnabled(False)
        self.control_layout.addWidget(self.try_fit_button, 2, 0)
        self.control_layout.addWidget(self.edit_parameter_button, 2, 1)
        self.result_group = QtWidgets.QGroupBox(self.tr("Result"))
        self.result_group.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
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
        self.chart_group.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
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
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self._dataset: Optional[Dataset] = None
        self._results: List[EMMAResult] = []

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
        kernel_type = self.SUPPORT_KERNELS[self.kernel_combo_box.currentIndex()][0]
        return kernel_type

    @property
    def n_members(self):
        return self.n_members_input.value()

    @property
    def n_results(self) -> int:
        return len(self._results)

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
            return self._results[self.selected_index]

    def on_dataset_loaded(self, dataset: Dataset):
        self._dataset = dataset
        self.try_fit_button.setEnabled(True)
        self.edit_parameter_button.setEnabled(True)

    def on_edit_parameter_clicked(self):
        if self._dataset is not None:
            target = np.mean(self._dataset.distributions, axis=0)
            target = target / np.sum(target)
            self.parameter_editor.setup_target(self._dataset.classes, target)
        self.parameter_editor.show()

    def on_try_fit_clicked(self):
        if self._dataset is None:
            self.logger.error("The dataset has not been loaded.")
            self.show_error(self.tr("The dataset has not been loaded."))
            return

        settings = {**self.setting_dialog.settings}
        if self.parameter_editor.parameter_enabled:
            self.logger.info("The parameters in Parameter Editor are enabled. They will be used preferentially!")
            settings["kernel_type"] = KernelType.__members__[self.parameter_editor.distribution_type.name]
            settings["n_members"] = self.parameter_editor.n_components
            settings["x0"] = self.parameter_editor.parameters[:-1, :].astype(np.float32)
        else:
            settings["kernel_type"] = self.kernel_type
            settings["n_members"] = self.n_members
        self.logger.debug(f"Try to perform the EMMA algorithm. Algorithm settings: {settings}.")

        if self._client is not None and self._client.has_target:
            try:
                server_state = self._client.get_service_state()
                if len(self._dataset) > server_state["max_dataset_size"]:
                    self.logger.error(
                        f"The dataset size ({len(self._dataset)} samples) exceeded the limitation of remote grpc server.")
                    self.show_error(self.tr("The dataset size exceeded the limitation of remote grpc server."))
                    return
                # TODO: async & progress report
                result_or_msg = self._client.get_emma_result(self._dataset, **settings)
                if isinstance(result_or_msg, EMMAResult):
                    self.logger.info(f"The result has been received from the remote grpc server {self._client._target}.")
                    self.add_results([result_or_msg])
                    self.result_chart.show_result(result_or_msg)
                    return
                else:
                    self.logger.error(f"The EMMA fitting task failed: {result_or_msg}.")
                    self.show_error(self.tr("The EMMA fitting task failed, please check the logs for more details."))
                    return
            except _InactiveRpcError:
                self.logger.warning("The remote grpc server is not available.")

        self.try_fit_button.setEnabled(False)
        self.edit_parameter_button.setEnabled(False)
        progress_dialog = QtWidgets.QProgressDialog(
            self.tr("Performing the EMMA algorithm..."), self.tr("Cancel"), 0, 100, self)
        progress_dialog.setWindowTitle("QGrain")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)

        def callback(progress: float):
            if progress_dialog.wasCanceled():
                raise StopIteration()
            progress_dialog.setValue(int(progress * 100))
            QtCore.QCoreApplication.processEvents()
        try:
            from ..emma import try_emma
            result = try_emma(self._dataset, **settings, progress_callback=callback)
            self.add_results([result])
            self.result_chart.show_result(result)
        except StopIteration:
            self.logger.info("The performing task was canceled.")
        finally:
            progress_dialog.close()
            self.try_fit_button.setEnabled(True)
            self.edit_parameter_button.setEnabled(True)

    @classmethod
    def get_result_name(cls, result: EMMAResult):
        return f"{result.dataset.name} ({result.n_members}, {result.kernel_type.name})"

    def add_results(self, results: List[EMMAResult]):
        if self.n_results == 0:
            self.remove_button.setEnabled(True)
            self.show_button.setEnabled(True)

        self._results.extend(results)
        self.result_list_widget.addItems([self.get_result_name(result) for result in results])

    def on_remove_clicked(self):
        if self.n_results == 0:
            return
        else:
            index = self.selected_index
            self._results.pop(index)
            self.result_list_widget.takeItem(index)

        if self.n_results == 0:
            self.remove_button.setEnabled(False)
            self.show_button.setEnabled(False)

    def on_show_clicked(self):
        result = self.selected_result
        if result is not None:
            self.result_chart.show_result(result)

    def load_result(self):
        filename, _ = self.file_dialog.getOpenFileName(
            self, self.tr("Choose the file which stores the dumped EMMA result"), ".", "Dumped EMMA Result (*.emma)")
        if filename is None or filename == "":
            return
        with open(filename, "rb") as f:
            result = pickle.load(f)
            if isinstance(result, EMMAResult):
                self.add_results([result])
                self.logger.info(f"The dumped EMMA result has been loaded.")
            else:
                self.logger.error("The binary file is invalid (i.e., the object in it is not the EMMA result).")
                self.show_error(self.tr("The binary file is invalid (i.e., the object in it is not the EMMA result)."))
                return

    def save_selected_result(self):
        if self.n_results == 0:
            self.logger.error("There is no EMMA result.")
            self.show_error(self.tr("There is no EMMA result."))
            return
        filename, _ = self.file_dialog.getSaveFileName(
            self, self.tr("Choose a filename to save the selected EMMA result"),
            ".", "Microsoft Excel (*.xlsx);;Dumped EMMA Result (*.emma)")
        if filename is None or filename == "":
            return
        # Excel
        if filename[-4:] == "xlsx":
            result = self.selected_result
            progress_dialog = QtWidgets.QProgressDialog(
                self.tr("Saving the EMMA result..."), self.tr("Cancel"), 0, 100, self)
            progress_dialog.setWindowTitle("QGrain")
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)

            def callback(progress: float):
                if progress_dialog.wasCanceled():
                    raise StopIteration()
                progress_dialog.setValue(int(progress * 100))
                QtCore.QCoreApplication.processEvents()
            try:
                save_emma(result, filename, progress_callback=callback, logger=self.logger)
            except StopIteration:
                self.logger.info("The saving task was canceled.")
            finally:
                progress_dialog.close()
        # Binary File
        else:
            with open(filename, "wb") as f:
                pickle.dump(self.selected_result, f)
                self.logger.info("The selected EMMA result has been dumped.")

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("EMMA Analyzer"))
        self.control_group.setTitle(self.tr("Control"))
        self.kernel_label.setText(self.tr("Kernel Type"))
        self.n_members_label.setText(self.tr("Number of End Members"))
        self.try_fit_button.setText(self.tr("Try Fit"))
        self.edit_parameter_button.setText(self.tr("Edit Parameters"))
        self.result_group.setTitle(self.tr("Result"))
        self.remove_button.setText(self.tr("Remove"))
        self.show_button.setText(self.tr("Show"))
        self.chart_group.setTitle(self.tr("Chart"))
