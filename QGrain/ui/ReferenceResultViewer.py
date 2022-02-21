__all__ = ["ReferenceResultViewer"]

import logging
import pickle
import time
import typing


from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QFileDialog, QGridLayout, QLabel, QMenu,
                               QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem)

from ..chart.DistanceCurveChart import DistanceCurveChart
from ..chart.MixedDistributionChart import MixedDistributionChart
from ..model import GrainSizeSample
from ..ssu import SSUResult, built_in_distances, get_distance_func_by_name


class ReferenceResultViewer(QDialog):
    PAGE_ROWS = 20
    logger = logging.getLogger("root.QGrain.ui.ReferenceResultViewer")
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("SSU Reference Result Viewer"))
        self.__fitting_results = []
        self.__reference_map = {}
        self.retry_tasks = {}
        self.init_ui()
        self.distance_chart = DistanceCurveChart(parent=self, toolbar=True)
        self.mixed_distribution_chart = MixedDistributionChart()
        self.file_dialog = QFileDialog(parent=self)
        self.update_page_list()
        self.update_page(self.page_index)

        self.remove_warning_msg = QMessageBox(self)
        self.remove_warning_msg.setStandardButtons(QMessageBox.No|QMessageBox.Yes)
        self.remove_warning_msg.setDefaultButton(QMessageBox.No)
        self.remove_warning_msg.setWindowTitle(self.tr("Warning"))
        self.remove_warning_msg.setText(self.tr("Are you sure to remove all SSU results?"))

        self.normal_msg = QMessageBox(self)

    def init_ui(self):
        self.data_table = QTableWidget(100, 100)
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.data_table, 0, 0, 1, 3)

        self.previous_button = QPushButton(self.tr("Previous"))
        self.previous_button.setToolTip(self.tr("Click to back to the previous page."))
        self.previous_button.clicked.connect(self.on_previous_button_clicked)
        self.current_page_combo_box = QComboBox()
        self.current_page_combo_box.addItem(self.tr("Page {0}").format(1))
        self.current_page_combo_box.currentIndexChanged.connect(self.update_page)
        self.next_button = QPushButton(self.tr("Next"))
        self.next_button.setToolTip(self.tr("Click to jump to the next page."))
        self.next_button.clicked.connect(self.on_next_button_clicked)
        self.main_layout.addWidget(self.previous_button, 1, 0)
        self.main_layout.addWidget(self.current_page_combo_box, 1, 1)
        self.main_layout.addWidget(self.next_button, 1, 2)

        self.distance_label = QLabel(self.tr("Distance"))
        self.distance_label.setToolTip(self.tr("It's the function to calculate the difference (on the contrary, similarity) between two samples."))
        self.distance_combo_box = QComboBox()
        self.distance_combo_box.addItems(built_in_distances)
        self.distance_combo_box.setCurrentText("log10MSE")
        self.distance_combo_box.currentTextChanged.connect(lambda: self.update_page(self.page_index))
        self.main_layout.addWidget(self.distance_label, 2, 0)
        self.main_layout.addWidget(self.distance_combo_box, 2, 1, 1, 2)
        self.menu = QMenu(self.data_table)
        self.mark_action = self.menu.addAction(self.tr("Mark Selection(s) as Reference"))
        self.mark_action.triggered.connect(self.mark_selections)
        self.unmark_action = self.menu.addAction(self.tr("Unmark Selection(s)"))
        self.unmark_action.triggered.connect(self.unmark_selections)
        self.remove_action = self.menu.addAction(self.tr("Remove Selection(s)"))
        self.remove_action.triggered.connect(self.remove_selections)
        self.remove_all_action = self.menu.addAction(self.tr("Remove All"))
        self.remove_all_action.triggered.connect(self.remove_all_results)
        self.plot_loss_chart_action = self.menu.addAction(self.tr("Plot Loss Chart"))
        self.plot_loss_chart_action.triggered.connect(self.show_distance)
        self.plot_distribution_chart_action = self.menu.addAction(self.tr("Plot Distribution Chart"))
        self.plot_distribution_chart_action.triggered.connect(self.show_distribution)
        self.plot_distribution_animation_action = self.menu.addAction(self.tr("Plot Distribution Chart (Animation)"))
        self.plot_distribution_animation_action.triggered.connect(self.show_history_distribution)

        self.load_dump_action = self.menu.addAction(self.tr("Load Binary Dump"))
        self.load_dump_action.triggered.connect(lambda: self.load_dump(mark_ref=True))
        self.save_dump_action = self.menu.addAction(self.tr("Save Binary Dump"))
        self.save_dump_action.triggered.connect(self.save_dump)
        self.data_table.customContextMenuRequested.connect(self.show_menu)

    def show_menu(self, pos):
        self.menu.popup(QCursor.pos())

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
    def distance_func(self) -> typing.Callable:
        return get_distance_func_by_name(self.distance_combo_box.currentText())

    @property
    def page_index(self) -> int:
        return self.current_page_combo_box.currentIndex()

    @property
    def n_pages(self) -> int:
        return self.current_page_combo_box.count()

    @property
    def n_results(self) -> int:
        return len(self.__fitting_results)

    @property
    def selections(self):
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
        self.current_page_combo_box.blockSignals(True)
        self.current_page_combo_box.clear()
        self.current_page_combo_box.addItems([self.tr("Page {0}").format(i+1) for i in range(n_pages)])
        if last_page_index >= n_pages:
            self.current_page_combo_box.setCurrentIndex(n_pages-1)
        else:
            self.current_page_combo_box.setCurrentIndex(last_page_index)
        self.current_page_combo_box.blockSignals(False)

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
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)
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
            self.tr("N_components"),
            self.tr("N_iterations"),
            self.tr("Spent Time [s]"),
            self.tr("Final Distance"),
            self.tr("Has Reference"),
            self.tr("Is Reference")])
        sample_names = [result.sample.name for result in self.__fitting_results[start: end]]
        self.data_table.setVerticalHeaderLabels(sample_names)
        for row, result in enumerate(self.__fitting_results[start: end]):
            write(row, 0, result.task.distribution_type.value)
            write(row, 1, result.task.n_components)
            write(row, 2, result.n_iterations)
            write(row, 3, result.time_spent)
            write(row, 4, self.distance_func(result.sample.distribution, result.distribution))
            has_ref = result.task.initial_guess is not None or result.task.reference is not None
            write(row, 5, self.tr("Yes") if has_ref else self.tr("No"))
            is_ref = result.uuid in self.__reference_map
            write(row, 6, self.tr("Yes") if is_ref else self.tr("No"))

        self.data_table.resizeColumnsToContents()

    def on_previous_button_clicked(self):
        if self.page_index > 0:
            self.current_page_combo_box.setCurrentIndex(self.page_index-1)

    def on_next_button_clicked(self):
        if self.page_index < self.n_pages - 1:
            self.current_page_combo_box.setCurrentIndex(self.page_index+1)

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
        if res == QMessageBox.Yes:
            self.__fitting_results.clear()
            self.update_page_list()
            self.update_page(0)

    def show_distance(self):
        results = [self.__fitting_results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.distance_chart.show_distance_series(
            result.get_distance_series(self.distance_name),
            title=result.sample.name)
        self.distance_chart.show()

    def show_distribution(self):
        results = [self.__fitting_results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.mixed_distribution_chart.show_model(result.view_model)
        self.mixed_distribution_chart.show()

    def show_history_distribution(self):
        results = [self.__fitting_results[i] for i in self.selections]
        if results is None or len(results) == 0:
            return
        result = results[0]
        self.mixed_distribution_chart.show_result(result)
        self.mixed_distribution_chart.show()

    def load_dump(self, mark_ref=False):
        filename, _ = self.file_dialog.getOpenFileName(
            self, self.tr("Select a binary dump file of SSU results"),
            None, self.tr("Binary dump (*.dump)"))
        if filename is None or filename == "":
            return
        with open(filename, "rb") as f:
            results = pickle.load(f)
            valid = True
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, SSUResult):
                        valid = False
                        break
            else:
                valid = False

            if valid:
                self.add_results(results)
                if mark_ref:
                    self.mark_results(results)
            else:
                self.show_error(self.tr("The binary dump file is invalid."))

    def save_dump(self):
        if self.n_results == 0:
            self.show_warning(self.tr("There is not any result in the list."))
            return
        filename, _  = self.file_dialog.getSaveFileName(self, self.tr("Save the SSU results to binary dump file"),
                                            None, self.tr("Binary dump (*.dump)"))
        if filename is None or filename == "":
            return
        with open(filename, "wb") as f:
            pickle.dump(self.__fitting_results, f)

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
            # TODO: To scale the classes of result to that of sample
            # use moments to calculate? MOMENTS MAY NOT BE PERFECT, MAY IGNORE THE MINOR DIFFERENCE
            # result_moments = logarithmic(result.classes_φ, result.distribution)
            # distance = sum([(sample_moments[key]-result_moments[key])**2 for key in keys_to_check])
            trans_dist = trans_func(result.classes_φ)
            distance = self.distance_func(result.distribution, trans_dist)

            if distance < min_distance:
                min_distance = distance
                min_result = result

        self.logger.debug(f"It took {time.time()-start_time:0.4f} s to query the reference from {len(ref_results)} results.")
        return min_result

    def query_reference(self, sample: GrainSizeSample):
        if len(self.__reference_map) == 0:
            self.logger.debug("No result is marked as reference.")
            return None
        return self.find_similar(sample, self.__reference_map.values())


if __name__ == "__main__":
    import sys

    from QGrain.entry import setup_app
    app, splash = setup_app()
    main = ReferenceResultViewer()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
