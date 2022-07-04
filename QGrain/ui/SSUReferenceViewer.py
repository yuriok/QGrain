__all__ = ["SSUReferenceViewer"]

import logging
import pickle
import time
import typing
from uuid import UUID

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..chart.DistanceCurveChart import DistanceCurveChart
from ..model import GrainSizeSample
from ..ssu import SSUResult, built_in_distances, get_distance_function


class SSUReferenceViewer(QtWidgets.QWidget):
    PAGE_ROWS = 20
    logger = logging.getLogger("QGrain.SSUReferenceViewer")
    result_displayed = QtCore.Signal(SSUResult, bool)
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.__fitting_results = [] # type: list[SSUResult]
        self.__reference_map = {} # type: dict[UUID, SSUResult]
        self.init_ui()
        self.distance_chart = DistanceCurveChart()
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self.update_page_list()
        self.update_page(self.page_index)
        self.remove_warning_msg = QtWidgets.QMessageBox(self)
        self.remove_warning_msg.setStandardButtons(QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes)
        self.remove_warning_msg.setDefaultButton(QtWidgets.QMessageBox.No)
        self.remove_warning_msg.setWindowTitle(self.tr("Warning"))
        self.remove_warning_msg.setText(self.tr("Are you sure to remove all SSU results?"))
        self.normal_msg = QtWidgets.QMessageBox(self)

    def init_ui(self):
        self.setWindowTitle(self.tr("SSU Reference Viewer"))
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
        self.menu = QtWidgets.QMenu(self.data_table)
        self.mark_action = self.menu.addAction(self.tr("Mark")) # type: QtGui.QAction
        self.mark_action.triggered.connect(self.mark_selections)
        self.unmark_action = self.menu.addAction(self.tr("Unmark")) # type: QtGui.QAction
        self.unmark_action.triggered.connect(self.unmark_selections)
        self.remove_action = self.menu.addAction(self.tr("Remove")) # type: QtGui.QAction
        self.remove_action.triggered.connect(self.remove_selections)
        self.remove_all_action = self.menu.addAction(self.tr("Remove All")) # type: QtGui.QAction
        self.remove_all_action.triggered.connect(self.remove_all_results)
        self.show_chart_action = self.menu.addAction(self.tr("Show Chart")) # type: QtGui.QAction
        self.show_chart_action.triggered.connect(self.show_chart)
        self.show_distance_action = self.menu.addAction(self.tr("Show Distance Series")) # type: QtGui.QAction
        self.show_distance_action.triggered.connect(self.show_distance)
        self.data_table.customContextMenuRequested.connect(self.show_menu)
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
        return len(self.__fitting_results)

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
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels([
            self.tr("Distribution Type"),
            self.tr("Number of Components"),
            self.tr("Number of Iterations"),
            self.tr("Spent Time [s]"),
            self.tr("Final Distance"),
            self.tr("Has Reference"),
            self.tr("Is Reference")])
        self.data_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap)
        sample_names = [result.sample.name for result in self.__fitting_results[start: end]]
        self.data_table.setVerticalHeaderLabels(sample_names)
        for row, result in enumerate(self.__fitting_results[start: end]):
            write(row, 0, result.task.distribution_type.value)
            write(row, 1, result.task.n_components)
            write(row, 2, result.n_iterations)
            write(row, 3, result.time_spent)
            write(row, 4, self.distance_function(result.sample.distribution, result.distribution))
            has_ref = result.task.initial_parameters is not None
            write(row, 5, self.tr("Yes") if has_ref else self.tr("No"))
            is_ref = result.uuid in self.__reference_map
            write(row, 6, self.tr("Yes") if is_ref else self.tr("No"))

        self.data_table.resizeColumnsToContents()

    def add_result(self, result: SSUResult):
        if self.n_results == 0 or \
            (self.page_index == self.n_pages - 1 and \
            divmod(self.n_results, self.PAGE_ROWS)[-1] != 0):
            need_update = True
        else:
            need_update = False
        self.__fitting_results.append(result)
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
        self.__fitting_results.extend(results)
        self.update_page_list()
        if need_update:
            self.update_page(self.page_index)

    def mark_results(self, results: typing.List[SSUResult]):
        for result in results:
            self.__reference_map[result.uuid] = result

        self.update_page(self.page_index)

    def unmark_results(self, results: typing.List[SSUResult]):
        for result in results:
            if result.uuid in self.__reference_map:
                self.__reference_map.pop(result.uuid)

        self.update_page(self.page_index)

    def add_references(self, results: typing.List[SSUResult]):
        self.add_results(results)
        self.mark_results(results)

    def mark_selections(self):
        results = [self.__fitting_results[selection] for selection in self.selections]
        self.mark_results(results)

    def unmark_selections(self):
        results = [self.__fitting_results[selection] for selection in self.selections]
        self.unmark_results(results)

    def remove_results(self, indexes):
        results = []
        for i in reversed(indexes):
            res = self.__fitting_results.pop(i)
            results.append(res)
        self.unmark_results(results)
        self.update_page_list()
        self.update_page(self.page_index)

    def remove_selections(self):
        indexes = self.selections
        self.remove_results(indexes)

    def remove_all_results(self):
        res = self.remove_warning_msg.exec_()
        if res == QtWidgets.QMessageBox.Yes:
            self.__fitting_results.clear()
            self.__reference_map.clear()
            self.update_page_list()
            self.update_page(0)

    def show_distance(self):
        results = [self.__fitting_results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.distance_chart.show_distance_series(
            result.get_distance_series(self.distance_name),
            ylabel=self.distance_name,
            title=result.sample.name)
        self.distance_chart.show()

    def show_chart(self):
        results = [self.__fitting_results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.result_displayed.emit(result)

    def load_references(self, mark_ref=False):
        filename, _ = self.file_dialog.getOpenFileName(
            self, self.tr("Choose a file which stores the dumped SSU results"),
            None, self.tr("Dumped SSU Results (*.ssu)"))
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
                    old_classes = self.__fitting_results[0].classes_φ
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
                if mark_ref:
                    self.mark_results(results)
                self.logger.info(f"There are {len(results)} SSU results that have been loaded.")
            else:
                self.logger.error("The binary file is invalid (i.e., the objects in it are not SSU results).")
                self.show_error(self.tr("The binary file is invalid (i.e., the objects in it are not SSU results)."))

    def dump_references(self):
        if self.n_results == 0:
            self.show_error(self.tr("There is no SSU result."))
            return
        filename, _  = self.file_dialog.getSaveFileName(
            self, self.tr("Choose a filename to dump the SSU results"),
            None, self.tr("Dumped SSU Results (*.ssu)"))
        if filename is None or filename == "":
            return
        with open(filename, "wb") as f:
            pickle.dump(self.__fitting_results, f)
        self.logger.info("The SSU results have been dumped.")

    def find_similar(self, target: GrainSizeSample, ref_results: typing.List[SSUResult]):
        assert len(ref_results) != 0
        # sample_moments = logarithmic(sample.classes_φ, sample.distribution)
        # keys_to_check = ["mean", "std", "skewness", "kurtosis"]

        start_time = time.time()
        from scipy.interpolate import interp1d
        min_distance = 1e100
        min_result = None
        trans_func = interp1d(target.classes_φ, target.distribution, bounds_error=False, fill_value=0.0)
        for result in ref_results:
            trans_dist = trans_func(result.classes_φ)
            distance = self.distance_function(result.distribution, trans_dist)
            if distance < min_distance:
                min_distance = distance
                min_result = result
        # self.logger.debug(f"It took {time.time()-start_time:0.4f} s to query the reference from {len(ref_results)} results.")
        return min_result

    def query_reference(self, sample: GrainSizeSample):
        if len(self.__reference_map) == 0:
            # self.logger.debug("Try to query the reference, but there is no result that has been marked as the reference.")
            return None
        return self.find_similar(sample, self.__reference_map.values())

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("SSU Reference Viewer"))
        self.previous_button.setText(self.tr("Previous"))
        self.previous_button.setToolTip(self.tr("Click to get back to the previous page."))
        for i in range(self.page_combo_box.count()):
            self.page_combo_box.setItemText(i, self.tr("Page {0}").format(i+1))
        self.next_button.setText(self.tr("Next"))
        self.next_button.setToolTip(self.tr("Click to jump to the next page."))
        self.distance_label.setText(self.tr("Distance Function"))
        self.distance_label.setToolTip(self.tr("The function to calculate the difference (on the contrary, similarity) between two samples."))
        self.mark_action.setText(self.tr("Mark"))
        self.unmark_action.setText(self.tr("Unmark"))
        self.remove_action.setText(self.tr("Remove"))
        self.remove_all_action.setText(self.tr("Remove All"))
        self.show_chart_action.setText(self.tr("Show Chart"))
        self.show_distance_action.setText(self.tr("Show Distance Series"))
        self.update_page(self.page_index)
