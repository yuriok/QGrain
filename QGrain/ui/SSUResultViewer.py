__all__ = ["SSUResultViewer"]

import logging
import pickle
from typing import *

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..models import SSUResult
from ..ssu import built_in_losses
from ..charts.BoxplotChart import BoxplotChart
from ..charts.LossSeriesChart import LossSeriesChart
from ..io import save_ssu
from .ParameterTable import ParameterTable


class SSUResultViewer(QtWidgets.QWidget):
    PAGE_ROWS = 20
    logger = logging.getLogger("QGrain.SSUResultViewer")
    result_displayed = QtCore.Signal(SSUResult)
    result_referred = QtCore.Signal(SSUResult)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._results: List[SSUResult] = []
        self.setWindowTitle(self.tr("SSU Result Viewer"))
        self.data_table = QtWidgets.QTableWidget(100, 100)
        self.data_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.addWidget(self.data_table, 0, 0, 1, 3)
        self.previous_button = QtWidgets.QPushButton(self.tr("Previous"))
        self.previous_button.setToolTip(self.tr("Click to get back to the previous page."))
        self.page_combo_box = QtWidgets.QComboBox()
        self.page_combo_box.addItem(self.tr("Page {0}").format(1))
        self.next_button = QtWidgets.QPushButton(self.tr("Next"))
        self.next_button.setToolTip(self.tr("Click to jump to the next page."))
        self.previous_button.clicked.connect(lambda: self.page_combo_box.setCurrentIndex(max(self.page_index-1, 0)))
        self.page_combo_box.currentIndexChanged.connect(self.update_page)
        self.next_button.clicked.connect(lambda: self.page_combo_box.setCurrentIndex(min(self.page_index+1, self.n_pages-1)))
        self.main_layout.addWidget(self.previous_button, 1, 0)
        self.main_layout.addWidget(self.page_combo_box, 1, 1)
        self.main_layout.addWidget(self.next_button, 1, 2)
        self.loss_label = QtWidgets.QLabel(self.tr("Loss"))
        self.loss_label.setToolTip(self.tr("The function to calculate the difference between prediction and observation."))
        self.loss_combo_box = QtWidgets.QComboBox()
        self.loss_combo_box.addItems(built_in_losses)
        self.loss_combo_box.setCurrentText("lmse")
        self.loss_combo_box.currentTextChanged.connect(lambda: self.update_page(self.page_index))
        self.main_layout.addWidget(self.loss_label, 2, 0)
        self.main_layout.addWidget(self.loss_combo_box, 2, 1, 1, 2)
        self.menu = QtWidgets.QMenu(self.data_table)
        self.menu.setShortcutAutoRepeat(True)
        self.remove_action = self.menu.addAction(self.tr("Remove"))
        self.remove_action.triggered.connect(self.remove_selections)
        self.remove_all_action = self.menu.addAction(self.tr("Remove All"))
        self.remove_all_action.triggered.connect(self.remove_all_results)
        self.refer_action = self.menu.addAction(self.tr("Refer Parameters"))
        self.refer_action.triggered.connect(self.refer_result)
        self.show_chart_action = self.menu.addAction(self.tr("Show Chart"))
        self.show_chart_action.triggered.connect(self.show_chart)
        self.auto_show_selected_action = self.menu.addAction(self.tr("Auto Show"))
        self.auto_show_selected_action.setCheckable(True)
        self.auto_show_selected_action.setChecked(False)
        self.show_distance_action = self.menu.addAction(self.tr("Show Loss Series"))
        self.show_distance_action.triggered.connect(self.show_loss_series)
        self.show_parameter_action = self.menu.addAction(self.tr("Show Parameters"))
        self.show_parameter_action.triggered.connect(self.show_parameters)
        self.detect_outliers_menu = self.menu.addMenu(self.tr("Check"))
        self.check_nan_and_inf_action = self.detect_outliers_menu.addAction(self.tr("NaN / Inf"))
        self.check_nan_and_inf_action.triggered.connect(self.check_nan_and_inf)
        self.check_final_distances_action = self.detect_outliers_menu.addAction(self.tr("Final Loss"))
        self.check_final_distances_action.triggered.connect(self.check_final_distances)
        self.check_mean_action = self.detect_outliers_menu.addAction(self.tr("Mean"))
        self.check_mean_action.triggered.connect(lambda: self.check_component_moments("mean"))
        self.check_std_action = self.detect_outliers_menu.addAction(self.tr("Sorting Coefficient"))
        self.check_std_action.triggered.connect(lambda: self.check_component_moments("std"))
        self.check_skewness_action = self.detect_outliers_menu.addAction(self.tr("Skewness"))
        self.check_skewness_action.triggered.connect(lambda: self.check_component_moments("skewness"))
        self.check_kurtosis_action = self.detect_outliers_menu.addAction(self.tr("Kurtosis"))
        self.check_kurtosis_action.triggered.connect(lambda: self.check_component_moments("kurtosis"))
        self.check_proportion_action = self.detect_outliers_menu.addAction(self.tr("Proportion"))
        self.check_proportion_action.triggered.connect(self.check_component_proportion)
        self.data_table.customContextMenuRequested.connect(self.show_menu)
        self.data_table.itemSelectionChanged.connect(self.on_selection_changed)
        # necessary to add actions of menu to this widget itself,
        # otherwise, the shortcuts will not be triggered
        self.addActions(self.menu.actions())

        self.boxplot_chart = BoxplotChart()
        self.loss_chart = LossSeriesChart()
        self.update_page_list()
        self.update_page(self.page_index)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.parameter_table = None

    def show_menu(self, pos: QtCore.QPoint):
        self.menu.popup(QtGui.QCursor.pos())

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
    def loss_name(self) -> str:
        return self.loss_combo_box.currentText()

    @property
    def page_index(self) -> int:
        return self.page_combo_box.currentIndex()

    @property
    def n_pages(self) -> int:
        return self.page_combo_box.count()

    @property
    def n_results(self) -> int:
        return len(self._results)

    @property
    def selections(self) -> List[int]:
        start = self.page_index*self.PAGE_ROWS
        temp = set()
        for item in self.data_table.selectedRanges():
            for i in range(item.topRow(), min(self.PAGE_ROWS+1, item.bottomRow()+1)):
                temp.add(i+start)
        indexes = list(temp)
        indexes.sort()
        return indexes

    @property
    def all_results(self) -> List[SSUResult]:
        return self._results.copy()

    @property
    def auto_show_selected(self) -> bool:
        return self.auto_show_selected_action.isChecked()

    def update_page_list(self):
        last_page_index = self.page_index
        if self.n_results == 0:
            n_pages = 1
        else:
            n_pages, left = divmod(self.n_results, self.PAGE_ROWS)
            if left != 0:
                n_pages += 1
        self.page_combo_box.blockSignals(True)
        self.page_combo_box.clear()
        self.page_combo_box.addItems([self.tr("Page {0}").format(i+1) for i in range(n_pages)])
        if last_page_index >= n_pages:
            self.page_combo_box.setCurrentIndex(n_pages-1)
        else:
            self.page_combo_box.setCurrentIndex(last_page_index)
        self.page_combo_box.blockSignals(False)

    def update_page(self, page_index: int):
        def write(row: int, col: int, value: Union[int, float, str]):
            if isinstance(value, str):
                pass
            elif isinstance(value, int):
                value = str(value)
            elif isinstance(value, float):
                value = f"{value: 0.4f}"
            else:
                value = value.__str__()
            item = QtWidgets.QTableWidgetItem(value)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.data_table.setItem(row, col, item)
        # necessary to clear
        self.data_table.clear()
        if page_index == self.n_pages - 1:
            start = page_index * self.PAGE_ROWS
            end = self.n_results
        else:
            start, end = page_index * self.PAGE_ROWS, (page_index+1) * self.PAGE_ROWS
        self.data_table.setRowCount(end-start)
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels([
            self.tr("Distribution Type"),
            self.tr("Number of Components"),
            self.tr("Number of Iterations"),
            self.tr("Spent Time [s]"),
            self.tr("Final Loss"),
            self.tr("Has Reference")])
        self.data_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap)
        sample_names = [result.sample.name for result in self._results[start: end]]
        self.data_table.setVerticalHeaderLabels(sample_names)
        for row, result in enumerate(self._results[start: end]):
            write(row, 0, result.distribution_type.name)
            write(row, 1, len(result))
            write(row, 2, result.n_iterations)
            write(row, 3, result.time_spent)
            write(row, 4, result.loss(self.loss_name))
            write(row, 5, self.tr("Yes") if result.x0 is not None else self.tr("No"))
        self.data_table.resizeColumnsToContents()

    def add_result(self, result: SSUResult):
        if self.n_results == 0 or (self.page_index==self.n_pages-1 and divmod(self.n_results, self.PAGE_ROWS)[-1]!=0):
            need_update = True
        else:
            need_update = False
        self._results.append(result)
        self.update_page_list()
        if need_update:
            self.update_page(self.page_index)

    def add_results(self, results: List[SSUResult]):
        if self.n_results == 0 or (self.page_index==self.n_pages-1 and divmod(self.n_results, self.PAGE_ROWS)[-1]!=0):
            need_update = True
        else:
            need_update = False
        self._results.extend(results)
        self.update_page_list()
        if need_update:
            self.update_page(self.page_index)

    def remove_results(self, indexes):
        results = []
        for i in reversed(indexes):
            res = self._results.pop(i)
            results.append(res)
        self.update_page_list()
        self.update_page(self.page_index)

    def remove_selections(self):
        indexes = self.selections
        self.remove_results(indexes)

    def remove_all_results(self):
        remove_msg = QtWidgets.QMessageBox(self)
        remove_msg.setStandardButtons(QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes)
        remove_msg.setDefaultButton(QtWidgets.QMessageBox.No)
        remove_msg.setWindowTitle(self.tr("Warning"))
        remove_msg.setText(self.tr("Are you sure to remove all SSU results?"))
        res = remove_msg.exec_()
        if res == QtWidgets.QMessageBox.Yes:
            self._results.clear()
            self.update_page_list()
            self.update_page(0)

    def refer_result(self):
        results = [self._results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.result_referred.emit(result)

    def show_chart(self):
        results = [self._results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.result_displayed.emit(result)

    def on_selection_changed(self):
        if self.auto_show_selected:
            self.show_chart()

    def show_loss_series(self):
        results = [self._results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        series = result.loss_series(self.loss_name)
        if len(series) == 1:
            self.show_warning(self.tr("The variation history was not recorded, can not calculate the loss series."))
            return
        self.loss_chart.show_loss_series(series, self.loss_name, result.sample.name)
        self.loss_chart.show()

    def show_parameters(self):
        results = [self._results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.parameter_table = ParameterTable(result)
        self.parameter_table.show()

    def load_results(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("Choose the file which stores the dumped SSU results"),
            ".", "Dumped SSU Results (*.ssu)")
        if filename is None or filename == "":
            return
        with open(filename, "rb") as f:
            results = pickle.load(f) # type: list[SSUResult]
            valid = True
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, SSUResult):
                        valid = False
                        break
            else:
                valid = False

            if valid:
                if self.n_results != 0 and len(results) != 0:
                    old_classes = self._results[0].classes_phi
                    new_classes = results[0].classes_phi
                    classes_inconsistent = False
                    if len(old_classes) != len(new_classes):
                        classes_inconsistent = True
                    else:
                        classes_error = np.abs(old_classes - new_classes)
                        if not np.all(np.less_equal(classes_error, 1e-8)):
                            classes_inconsistent = True
                    if classes_inconsistent:
                        self.logger.error("The grain size classes of the SSU results in binary file are inconsistent with that in list.")
                        self.show_error(self.tr("The grain size classes of the SSU results in binary file are inconsistent with that in list."))
                        return
                self.add_results(results)
                self.logger.info(f"There are {len(results)} SSU results that have been loaded.")
            else:
                self.logger.error("The binary file is invalid (i.e., the objects in it are not SSU results).")
                self.show_error(self.tr("The binary file is invalid (i.e., the objects in it are not SSU results)."))

    def save_results(self, align_components=False):
        if self.n_results == 0:
            self.show_error(self.tr("There is no SSU result."))
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.tr("Choose a filename to save the SSU Results"),
            ".", "Microsoft Excel (*.xlsx);;Dumped SSU Results (*.ssu)")
        if filename is None or filename == "":
            return
            # Excel
        if filename[-4:] == "xlsx":
            progress_dialog = QtWidgets.QProgressDialog(
                    self.tr("Saving the SSU results..."), self.tr("Cancel"),
                    0, 100, self)
            progress_dialog.setWindowTitle("QGrain")
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)

            def callback(progress: float):
                if progress_dialog.wasCanceled():
                    raise StopIteration()
                progress_dialog.setValue(int(progress*100))
                QtCore.QCoreApplication.processEvents()
            try:
                save_ssu(self._results, filename, align_components, progress_callback=callback, logger=self.logger)
            except StopIteration as e:
                self.logger.info("The saving task was canceled.")
            finally:
                progress_dialog.close()
        else:
            with open(filename, "wb") as f:
                pickle.dump(self._results, f)
                self.logger.info("All SSU results have been dumped.")

    def ask_deal_outliers(self, outlier_results: List[SSUResult], outlier_indexes: List[int]):
        assert len(outlier_indexes) == len(outlier_results)
        if len(outlier_results) == 0:
            self.logger.info("There is no result was evaluated as the outlier.")
        else:
            self.logger.info(f"There are {len(outlier_results)} results were evaluated as the outliers: {', '.join([result.sample.name for result in outlier_results])}.")
            outlier_msg = QtWidgets.QMessageBox(self)
            outlier_msg.setStandardButtons(QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes)
            outlier_msg.setDefaultButton(QtWidgets.QMessageBox.No)
            outlier_msg.setText(self.tr("There are {0} results were evaluated as the outliers. Please check the logs for more details. Do you want to remove them?").format(len(outlier_results)))
            res = outlier_msg.exec_()
            if res == QtWidgets.QMessageBox.Yes:
                self.remove_results(outlier_indexes)
                self.logger.debug("The outliers have been removed.")

    def check_nan_and_inf(self):
        if self.n_results == 0:
            self.show_error(self.tr("There is no SSU result."))
            return
        outlier_results = []
        outlier_indexes = []
        for i, result in enumerate(self._results):
            if not result.is_valid:
                outlier_results.append(result)
                outlier_indexes.append(i)
        self.logger.debug("Check if there is any NaN or Inf value.")
        self.ask_deal_outliers(outlier_results, outlier_indexes)

    def check_final_distances(self):
        if self.n_results == 0:
            self.show_error(self.tr("There is no SSU result."))
            return
        elif self.n_results < 10:
            self.show_error(self.tr("The number of results is not enough."))
            return
        losses = []
        for result in self._results:
            losses.append(result.loss(self.loss_name))
        losses = np.array(losses)
        self.boxplot_chart.show_dataset([losses], xlabels=[self.loss_name], ylabel="Loss")
        self.boxplot_chart.show()

        # calculate the 1/4, 1/2, and 3/4 position value to judge which result is invalid
        # 1. the mean squared errors are much higher in the results which are lack of components
        # 2. with the component number getting higher, the mean squared error will get lower and finally reach the minimum
        median = np.median(losses)
        upper_group = losses[np.greater(losses, median)]
        lower_group = losses[np.less(losses, median)]
        value_1_4 = np.median(lower_group)
        value_3_4 = np.median(upper_group)
        loss_QR = value_3_4 - value_1_4
        outlier_results = []
        outlier_indexes = []
        for i, (result, loss) in enumerate(zip(self._results, losses)):
            if loss > value_3_4 + loss_QR * 1.5:
            # which error too small is not outlier
            # if loss > value_3_4 + loss_QR * 1.5 or loss < value_1_4 - loss_QR * 1.5:
                outlier_results.append(result)
                outlier_indexes.append(i)
        self.logger.debug(f"Check the final losses using Whisker plot.")
        self.ask_deal_outliers(outlier_results, outlier_indexes)

    def check_component_moments(self, key: str):
        if self.n_results == 0:
            self.show_error(self.tr("There is no SSU result."))
            return
        elif self.n_results < 10:
            self.show_error(self.tr("The number of results is not enough."))
            return
        max_n_components = 0
        for result in self._results:
            if len(result) > max_n_components:
                max_n_components = len(result)
        moments = []
        for i in range(max_n_components):
            moments.append([])

        for result in self._results:
            for i, component in enumerate(result):
                if np.isnan(component.moments[key]) or np.isinf(component.moments[key]):
                    pass
                else:
                    moments[i].append(component.moments[key])

        key_label_trans = {"mean": "Mean [φ]", "std": "Sorting Coefficient", "skewness": "Skewness", "kurtosis": "Kurtosis"}
        self.boxplot_chart.show_dataset(moments, xlabels=[f"C{i+1}" for i in range(max_n_components)], ylabel=key_label_trans[key])
        self.boxplot_chart.show()

        outlier_dict = {}

        for i in range(max_n_components):
            stacked_moments = np.array(moments[i])
            # calculate the 1/4, 1/2, and 3/4 position value to judge which result is invalid
            # 1. the mean squared errors are much higher in the results which are lack of components
            # 2. with the component number getting higher, the mean squared error will get lower and finally reach the minimum
            median = np.median(stacked_moments)
            upper_group = stacked_moments[np.greater(stacked_moments, median)]
            lower_group = stacked_moments[np.less(stacked_moments, median)]
            value_1_4 = np.median(lower_group)
            value_3_4 = np.median(upper_group)
            moment_QR = value_3_4 - value_1_4

            for j, result in enumerate(self._results):
                if len(result) > i:
                    moment = result[i].moments[key]
                    if moment > value_3_4 + moment_QR * 1.5 or moment < value_1_4 - moment_QR * 1.5:
                        outlier_dict[j] = result

        outlier_results = []
        outlier_indexes = []
        for index, result in sorted(outlier_dict.items(), key=lambda x:x[0]):
            outlier_indexes.append(index)
            outlier_results.append(result)
        self.logger.debug(f"Check the {key_label_trans[key]} values using Whisker plot.")
        self.ask_deal_outliers(outlier_results, outlier_indexes)

    def check_component_proportion(self):
        outlier_results = []
        outlier_indexes = []
        for i, result in enumerate(self._results):
            for component in result:
                if component.proportion < 1e-3:
                    outlier_results.append(result)
                    outlier_indexes.append(i)
                    break
        self.logger.debug("Check if the proportion of any component is near zero.")
        self.ask_deal_outliers(outlier_results, outlier_indexes)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Up:
            self.page_combo_box.setCurrentIndex(max(self.page_index-1, 0))
        elif event.key() == QtCore.Qt.Key_Down:
            self.page_combo_box.setCurrentIndex(min(self.page_index+1, self.n_pages-1))

    def retranslate(self):
        self.setWindowTitle(self.tr("SSU Result Viewer"))
        self.previous_button.setText(self.tr("Previous"))
        self.previous_button.setToolTip(self.tr("Click to get back to the previous page."))
        for i in range(self.page_combo_box.count()):
            self.page_combo_box.setItemText(i, self.tr("Page {0}").format(i+1))
        self.next_button.setText(self.tr("Next"))
        self.next_button.setToolTip(self.tr("Click to jump to the next page."))
        self.loss_label.setText(self.tr("Loss"))
        self.loss_label.setToolTip(self.tr("The function to calculate the difference between prediction and observation."))
        self.remove_action.setText(self.tr("Remove"))
        self.remove_all_action.setText(self.tr("Remove All"))
        self.refer_action.setText(self.tr("Refer Parameters"))
        self.show_chart_action.setText(self.tr("Show Chart"))
        self.auto_show_selected_action.setText(self.tr("Auto Show"))
        self.show_distance_action.setText(self.tr("Show Loss Series"))
        self.show_parameter_action.setText(self.tr("Show Parameters"))
        self.detect_outliers_menu.setTitle(self.tr("Check"))
        self.check_nan_and_inf_action.setText(self.tr("NaN / Inf"))
        self.check_final_distances_action.setText(self.tr("Final Loss"))
        self.check_mean_action.setText(self.tr("Mean"))
        self.check_std_action.setText(self.tr("Sorting Coefficient"))
        self.check_skewness_action.setText(self.tr("Skewness"))
        self.check_kurtosis_action.setText(self.tr("Kurtosis"))
        self.check_proportion_action.setText(self.tr("Proportion"))
        self.update_page(self.page_index)
