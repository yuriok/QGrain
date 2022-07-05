__all__ = ["SSUResultViewer"]

import logging
import pickle
import typing

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..chart.BoxplotChart import BoxplotChart
from ..chart.DistanceCurveChart import DistanceCurveChart
from ..io import save_ssu
from ..ssu import SSUResult, built_in_distances, get_distance_function
from .ParameterTable import ParameterTable


class SSUResultViewer(QtWidgets.QWidget):
    PAGE_ROWS = 20
    logger = logging.getLogger("QGrain.SSUResultViewer")
    result_marked = QtCore.Signal(SSUResult)
    result_displayed = QtCore.Signal(SSUResult)
    result_referred = QtCore.Signal(SSUResult)
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.__results = [] # type: list[SSUResult]
        self.init_ui()
        self.boxplot_chart = BoxplotChart()
        self.distance_chart = DistanceCurveChart()
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self.update_page_list()
        self.update_page(self.page_index)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.parameter_table = None

    def init_ui(self):
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

        self.distance_label = QtWidgets.QLabel(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The function to calculate the difference (on the contrary, similarity) between two samples."))
        self.distance_combo_box = QtWidgets.QComboBox()
        self.distance_combo_box.addItems(built_in_distances)
        self.distance_combo_box.setCurrentText("log10MSE")
        self.distance_combo_box.currentTextChanged.connect(lambda: self.update_page(self.page_index))
        self.main_layout.addWidget(self.distance_label, 2, 0)
        self.main_layout.addWidget(self.distance_combo_box, 2, 1, 1, 2)
        self.menu = QtWidgets.QMenu(self.data_table) # type: QtWidgets.QMenu
        self.menu.setShortcutAutoRepeat(True)
        self.remove_action = self.menu.addAction(self.tr("Remove")) # type: QtGui.QAction
        self.remove_action.triggered.connect(self.remove_selections)
        self.remove_all_action = self.menu.addAction(self.tr("Remove All")) # type: QtGui.QAction
        self.remove_all_action.triggered.connect(self.remove_all_results)
        self.mark_action = self.menu.addAction(self.tr("Mark Reference")) # type: QtGui.QAction
        self.mark_action.triggered.connect(self.mark_selections)
        self.refer_action = self.menu.addAction(self.tr("Refer Parameters")) # type: QtGui.QAction
        self.refer_action.triggered.connect(self.refer_result)
        self.show_chart_action = self.menu.addAction(self.tr("Show Chart")) # type: QtGui.QAction
        self.show_chart_action.triggered.connect(self.show_chart)
        self.auto_show_selected_action = self.menu.addAction(self.tr("Auto Show")) # type: QtGui.QAction
        self.auto_show_selected_action.setCheckable(True)
        self.auto_show_selected_action.setChecked(False)
        self.show_distance_action = self.menu.addAction(self.tr("Show Distance Series")) # type: QtGui.QAction
        self.show_distance_action.triggered.connect(self.show_distance_series)
        self.show_parameter_action = self.menu.addAction(self.tr("Show Parameters")) # type: QtGui.QAction
        self.show_parameter_action.triggered.connect(self.show_parameters)
        self.detect_outliers_menu = self.menu.addMenu(self.tr("Check")) # type: QtWidgets.QMenu
        self.check_nan_and_inf_action = self.detect_outliers_menu.addAction(self.tr("NaN / Inf")) # type: QtGui.QAction
        self.check_nan_and_inf_action.triggered.connect(self.check_nan_and_inf)
        self.check_final_distances_action = self.detect_outliers_menu.addAction(self.tr("Final Distance")) # type: QtGui.QAction
        self.check_final_distances_action.triggered.connect(self.check_final_distances)
        self.check_mean_action = self.detect_outliers_menu.addAction(self.tr("Mean")) # type: QtGui.QAction
        self.check_mean_action.triggered.connect(lambda: self.check_component_moments("mean"))
        self.check_std_action = self.detect_outliers_menu.addAction(self.tr("Sorting Coefficient")) # type: QtGui.QAction
        self.check_std_action.triggered.connect(lambda: self.check_component_moments("std"))
        self.check_skewness_action = self.detect_outliers_menu.addAction(self.tr("Skewness")) # type: QtGui.QAction
        self.check_skewness_action.triggered.connect(lambda: self.check_component_moments("skewness"))
        self.check_kurtosis_action = self.detect_outliers_menu.addAction(self.tr("Kurtosis")) # type: QtGui.QAction
        self.check_kurtosis_action.triggered.connect(lambda: self.check_component_moments("kurtosis"))
        self.check_proportion_action = self.detect_outliers_menu.addAction(self.tr("Proportion")) # type: QtGui.QAction
        self.check_proportion_action.triggered.connect(self.check_component_proportion)
        self.data_table.customContextMenuRequested.connect(self.show_menu)
        self.data_table.itemSelectionChanged.connect(self.on_selection_changed)
        # necessary to add actions of menu to this widget itself,
        # otherwise, the shortcuts will not be triggered
        self.addActions(self.menu.actions())

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
    def distance_name(self) -> str:
        return self.distance_combo_box.currentText()

    @property
    def distance_function(self) -> typing.Callable:
        return get_distance_function(self.distance_combo_box.currentText())

    @property
    def page_index(self) -> int:
        return self.page_combo_box.currentIndex()

    @property
    def n_pages(self) -> int:
        return self.page_combo_box.count()

    @property
    def n_results(self) -> int:
        return len(self.__results)

    @property
    def selections(self) -> typing.List[int]:
        start = self.page_index*self.PAGE_ROWS
        temp = set()
        for item in self.data_table.selectedRanges():
            for i in range(item.topRow(), min(self.PAGE_ROWS+1, item.bottomRow()+1)):
                temp.add(i+start)
        indexes = list(temp)
        indexes.sort()
        return indexes

    @property
    def all_results(self) -> typing.List[SSUResult]:
        return self.__results.copy()

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
        def write(row: int, col: int, value: str):
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
            self.tr("Final Distance"),
            self.tr("Has Reference")])
        self.data_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap)
        sample_names = [result.sample.name for result in self.__results[start: end]]
        self.data_table.setVerticalHeaderLabels(sample_names)
        for row, result in enumerate(self.__results[start: end]):
            write(row, 0, result.task.distribution_type.value)
            write(row, 1, result.task.n_components)
            write(row, 2, result.n_iterations)
            write(row, 3, result.time_spent)
            write(row, 4, self.distance_function(result.sample.distribution, result.distribution))
            has_ref = result.task.initial_parameters is not None
            write(row, 5, self.tr("Yes") if has_ref else self.tr("No"))

        self.data_table.resizeColumnsToContents()

    def add_result(self, result: SSUResult):
        if self.n_results == 0 or \
            (self.page_index == self.n_pages - 1 and \
            divmod(self.n_results, self.PAGE_ROWS)[-1] != 0):
            need_update = True
        else:
            need_update = False
        self.__results.append(result)
        self.update_page_list()
        if need_update:
            self.update_page(self.page_index)

    def add_results(self, results: typing.List[SSUResult]):
        if self.n_results == 0 or \
            (self.page_index == self.n_pages - 1 and \
            divmod(self.n_results, self.PAGE_ROWS)[-1] != 0):
            need_update = True
        else:
            need_update = False
        self.__results.extend(results)
        self.update_page_list()
        if need_update:
            self.update_page(self.page_index)

    def remove_results(self, indexes):
        results = []
        for i in reversed(indexes):
            res = self.__results.pop(i)
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
            self.__results.clear()
            self.update_page_list()
            self.update_page(0)

    def mark_selections(self):
        for index in self.selections:
            self.result_marked.emit(self.__results[index])

    def refer_result(self):
        results = [self.__results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.result_referred.emit(result)

    def show_chart(self):
        results = [self.__results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.result_displayed.emit(result)

    def on_selection_changed(self):
        if self.auto_show_selected:
            self.show_chart()

    def show_distance_series(self):
        results = [self.__results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.distance_chart.show_distance_series(
            result.get_distance_series(self.distance_name),
            self.distance_name,
            result.sample.name)
        self.distance_chart.show()

    def show_parameters(self):
        results = [self.__results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.parameter_table = ParameterTable(result)
        self.parameter_table.show()

    def load_results(self):
        filename, _ = self.file_dialog.getOpenFileName(
            self, self.tr("Choose the file which stores the dumped SSU results"),
            None, "Dumped SSU Results (*.ssu)")
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
                    old_classes = self.__results[0].classes_φ
                    new_classes = results[0].classes_φ
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
        filename, _ = self.file_dialog.getSaveFileName(
            None, self.tr("Choose a filename to save the SSU Results"),
            None, "Microsoft Excel (*.xlsx);;Dumped SSU Results (*.ssu)")
        if filename is None or filename == "":
            return
        try:
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
                save_ssu(self.__results, filename, align_components, progress_callback=callback, logger=self.logger)
            else:
                with open(filename, "wb") as f:
                    pickle.dump(self.__results, f)
                    self.logger.info("All SSU results have been dumped.")

        except Exception as e:
            self.logger.exception("An unknown exception was raised. Please check the logs for more details.", stack_info=True)
            self.show_error(self.tr("An unknown exception was raised. Please check the logs for more details."))

    def ask_deal_outliers(self, outlier_results: typing.List[SSUResult],
                          outlier_indexes: typing.List[int]):
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
        for i, result in enumerate(self.__results):
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
        distances = []
        for result in self.__results:
            distances.append(result.get_distance(self.distance_name))
        distances = np.array(distances)
        self.boxplot_chart.show_dataset([distances], xlabels=[self.distance_name], ylabel="Distance")
        self.boxplot_chart.show()

        # calculate the 1/4, 1/2, and 3/4 postion value to judge which result is invalid
        # 1. the mean squared errors are much higher in the results which are lack of components
        # 2. with the component number getting higher, the mean squared error will get lower and finally reach the minimum
        median = np.median(distances)
        upper_group = distances[np.greater(distances, median)]
        lower_group = distances[np.less(distances, median)]
        value_1_4 = np.median(lower_group)
        value_3_4 = np.median(upper_group)
        distance_QR = value_3_4 - value_1_4
        outlier_results = []
        outlier_indexes = []
        for i, (result, distance) in enumerate(zip(self.__results, distances)):
            if distance > value_3_4 + distance_QR * 1.5:
            # which error too small is not outlier
            # if distance > value_3_4 + distance_QR * 1.5 or distance < value_1_4 - distance_QR * 1.5:
                outlier_results.append(result)
                outlier_indexes.append(i)
        self.logger.debug(f"Check the final distances using Whisker plot.")
        self.ask_deal_outliers(outlier_results, outlier_indexes)

    def check_component_moments(self, key: str):
        if self.n_results == 0:
            self.show_error(self.tr("There is no SSU result."))
            return
        elif self.n_results < 10:
            self.show_error(self.tr("The number of results is not enough."))
            return
        max_n_components = 0
        for result in self.__results:
            if result.n_components > max_n_components:
                max_n_components = result.n_components
        moments = []
        for i in range(max_n_components):
            moments.append([])

        for result in self.__results:
            for i, component in enumerate(result.components):
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
            # calculate the 1/4, 1/2, and 3/4 postion value to judge which result is invalid
            # 1. the mean squared errors are much higher in the results which are lack of components
            # 2. with the component number getting higher, the mean squared error will get lower and finally reach the minimum
            median = np.median(stacked_moments)
            upper_group = stacked_moments[np.greater(stacked_moments, median)]
            lower_group = stacked_moments[np.less(stacked_moments, median)]
            value_1_4 = np.median(lower_group)
            value_3_4 = np.median(upper_group)
            distance_QR = value_3_4 - value_1_4

            for j, result in enumerate(self.__results):
                if result.n_components > i:
                    distance = result.components[i].moments[key]
                    if distance > value_3_4 + distance_QR * 1.5 or distance < value_1_4 - distance_QR * 1.5:
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
        for i, result in enumerate(self.__results):
            for component in result.components:
                if component.proportion < 1e-3:
                    outlier_results.append(result)
                    outlier_indexes.append(i)
                    break
        self.logger.debug("Check if the proportion of any component is near zero.")
        self.ask_deal_outliers(outlier_results, outlier_indexes)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("SSU Result Viewer"))
        self.previous_button.setText(self.tr("Previous"))
        self.previous_button.setToolTip(self.tr("Click to get back to the previous page."))
        for i in range(self.page_combo_box.count()):
            self.page_combo_box.setItemText(i, self.tr("Page {0}").format(i+1))
        self.next_button.setText(self.tr("Next"))
        self.next_button.setToolTip(self.tr("Click to jump to the next page."))
        self.distance_label.setText(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The function to calculate the difference (on the contrary, similarity) between two samples."))
        self.remove_action.setText(self.tr("Remove"))
        self.remove_all_action.setText(self.tr("Remove All"))
        self.mark_action.setText(self.tr("Mark Reference"))
        self.refer_action.setText(self.tr("Refer Parameters"))
        self.show_chart_action.setText(self.tr("Show Chart"))
        self.auto_show_selected_action.setText(self.tr("Auto Show"))
        self.show_distance_action.setText(self.tr("Show Distance Series"))
        self.show_parameter_action.setText(self.tr("Show Parameters"))
        self.detect_outliers_menu.setTitle(self.tr("Check"))
        self.check_nan_and_inf_action.setText(self.tr("NaN / Inf"))
        self.check_final_distances_action.setText(self.tr("Final Distance"))
        self.check_mean_action.setText(self.tr("Mean"))
        self.check_std_action.setText(self.tr("Sorting Coefficient"))
        self.check_skewness_action.setText(self.tr("Skewness"))
        self.check_kurtosis_action.setText(self.tr("Kurtosis"))
        self.check_proportion_action.setText(self.tr("Proportion"))
        self.update_page(self.page_index)
