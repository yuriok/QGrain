__all__ = ["StatisticalAnalyzer"]

from typing import *

from PySide6 import QtCore, QtGui, QtWidgets

from ..statistics import all_statistics
from ..models import Dataset, Sample
from ..charts import (highlight_color, FrequencyChart, Frequency3DChart, FrequencyHeatmap, CumulativeChart,
                      DiagramChart, BP12SSCDiagramChart, BP12GSMDiagramChart, Folk54SSCDiagramChart,
                      Folk54GSMDiagramChart, CMDiagramChart)


class StatisticalAnalyzer(QtWidgets.QWidget):
    PAGE_ROWS = 20

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._dataset: Optional[Dataset] = None
        self.data_table = QtWidgets.QTableWidget(100, 100)
        self.data_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.data_table.setHorizontalHeaderLabels([self.tr("Tips")])
        for row, tip in enumerate(self.tips):
            item = QtWidgets.QTableWidgetItem(tip)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.data_table.setItem(row, 0, item)
        self.data_table.setColumnWidth(0, 800)
        self.data_table.setColumnCount(1)
        self.data_table.setRowCount(len(self.tips))
        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.addWidget(self.data_table, 0, 0, 1, 3)
        self.previous_button = QtWidgets.QPushButton(self.tr("Previous"))
        self.previous_button.setToolTip(self.tr("Click to back to the previous page."))
        self.previous_button.clicked.connect(self.on_previous_button_clicked)
        self.current_page_combo_box = QtWidgets.QComboBox()
        self.current_page_combo_box.addItem(self.tr("No Page"))
        _ = self.tr("Page {0}").format(1)
        self.current_page_combo_box.currentIndexChanged.connect(self.update_page)
        self.next_button = QtWidgets.QPushButton(self.tr("Next"))
        self.next_button.setToolTip(self.tr("Click to jump to the next page."))
        self.next_button.clicked.connect(self.on_next_button_clicked)
        self.main_layout.addWidget(self.previous_button, 1, 0)
        self.main_layout.addWidget(self.current_page_combo_box, 1, 1)
        self.main_layout.addWidget(self.next_button, 1, 2)
        self.geometric_checkbox = QtWidgets.QCheckBox(self.tr("Geometric [unit is {0}]").format("μm"))
        self.geometric_checkbox.setChecked(True)
        self.geometric_checkbox.stateChanged.connect(self.on_is_geometric_changed)
        self.main_layout.addWidget(self.geometric_checkbox, 2, 0)
        self.FW57_checkbox = QtWidgets.QCheckBox(self.tr("Method of Statistical Moments"))
        self.FW57_checkbox.setChecked(False)
        self.FW57_checkbox.stateChanged.connect(self.on_is_FW57_changed)
        self.main_layout.addWidget(self.FW57_checkbox, 2, 1)
        self.proportion_combo_box = QtWidgets.QComboBox()
        self.proportion_combo_box.addItems([description for _, description in self.supported_proportions])
        self.proportion_combo_box.currentIndexChanged.connect(lambda: self.update_page(self.page_index))
        self.main_layout.addWidget(self.proportion_combo_box, 2, 2)
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setColumnStretch(1, 1)
        self.main_layout.setColumnStretch(2, 1)
        self.menu = QtWidgets.QMenu(self.data_table)
        self.plot_cumulative_menu = self.menu.addMenu(self.tr("Plot Cumulative Frequency Chart"))
        self.cumulative_plot_selected_action = self.plot_cumulative_menu.addAction(self.tr("Plot"))
        self.cumulative_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.cumulative_chart, self.selections, False))
        self.cumulative_append_selected_action = self.plot_cumulative_menu.addAction(self.tr("Append"))
        self.cumulative_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.cumulative_chart, self.selections, True))
        self.cumulative_plot_all_action = self.plot_cumulative_menu.addAction(self.tr("Plot All"))
        self.cumulative_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.cumulative_chart, self._dataset, False))
        self.cumulative_append_all_action = self.plot_cumulative_menu.addAction(self.tr("Append All"))
        self.cumulative_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.cumulative_chart, self._dataset, True))
        self.plot_frequency_menu = self.menu.addMenu(self.tr("Plot Frequency Distribution Chart"))
        self.frequency_plot_selected_action = self.plot_frequency_menu.addAction(self.tr("Plot"))
        self.frequency_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_chart, self.selections, False))
        self.frequency_append_selected_action = self.plot_frequency_menu.addAction(self.tr("Append"))
        self.frequency_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_chart, self.selections, True))
        self.frequency_plot_all_action = self.plot_frequency_menu.addAction(self.tr("Plot All"))
        self.frequency_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_chart, self._dataset, False))
        self.frequency_append_all_action = self.plot_frequency_menu.addAction(self.tr("Append All"))
        self.frequency_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_chart, self._dataset, True))
        self.plot_frequency_3D_menu = self.menu.addMenu(self.tr("Plot Frequency 3D Chart"))
        self.frequency_3D_plot_selected_action = self.plot_frequency_3D_menu.addAction(self.tr("Plot"))
        self.frequency_3D_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_3D_chart, self.selections, False))
        self.frequency_3D_append_selected_action = self.plot_frequency_3D_menu.addAction(self.tr("Append"))
        self.frequency_3D_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_3D_chart, self.selections, True))
        self.frequency_3D_plot_all_action = self.plot_frequency_3D_menu.addAction(self.tr("Plot All"))
        self.frequency_3D_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_3D_chart, self._dataset, False))
        self.frequency_3D_append_all_action = self.plot_frequency_3D_menu.addAction(self.tr("Append All"))
        self.frequency_3D_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_3D_chart, self._dataset, True))
        self.plot_frequency_heatmap_menu = self.menu.addMenu(self.tr("Plot Frequency Heatmap"))
        self.frequency_heatmap_plot_selected_action = self.plot_frequency_heatmap_menu.addAction(self.tr("Plot"))
        self.frequency_heatmap_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_heatmap, self.selections, False))
        self.frequency_heatmap_append_selected_action = self.plot_frequency_heatmap_menu.addAction(self.tr("Append"))
        self.frequency_heatmap_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_heatmap, self.selections, True))
        self.frequency_heatmap_plot_all_action = self.plot_frequency_heatmap_menu.addAction(self.tr("Plot All"))
        self.frequency_heatmap_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_heatmap, self._dataset, False))
        self.frequency_heatmap_append_all_action = self.plot_frequency_heatmap_menu.addAction(self.tr("Append All"))
        self.frequency_heatmap_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.frequency_heatmap, self._dataset, True))
        self.folk54_GSM_menu = self.menu.addMenu(self.tr("Plot GSM Diagram (Folk, 1954)"))
        self.folk54_GSM_plot_selected_action = self.folk54_GSM_menu.addAction(self.tr("Plot"))
        self.folk54_GSM_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.folk54_GSM_diagram, self.selections, False))
        self.folk54_GSM_append_selected_action = self.folk54_GSM_menu.addAction(self.tr("Append"))
        self.folk54_GSM_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.folk54_GSM_diagram, self.selections, True))
        self.folk54_GSM_plot_all_action = self.folk54_GSM_menu.addAction(self.tr("Plot All"))
        self.folk54_GSM_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.folk54_GSM_diagram, self._dataset, False))
        self.folk54_GSM_append_all_action = self.folk54_GSM_menu.addAction(self.tr("Append All"))
        self.folk54_GSM_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.folk54_GSM_diagram, self._dataset, True))
        self.folk54_SSC_menu = self.menu.addMenu(self.tr("Plot SSC Diagram (Folk, 1954)"))
        self.folk54_SSC_plot_selected_action = self.folk54_SSC_menu.addAction(self.tr("Plot"))
        self.folk54_SSC_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.folk54_SSC_diagram, self.selections, False))
        self.folk54_SSC_append_selected_action = self.folk54_SSC_menu.addAction(self.tr("Append"))
        self.folk54_SSC_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.folk54_SSC_diagram, self.selections, True))
        self.folk54_SSC_plot_all_action = self.folk54_SSC_menu.addAction(self.tr("Plot All"))
        self.folk54_SSC_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.folk54_SSC_diagram, self._dataset, False))
        self.folk54_SSC_append_all_action = self.folk54_SSC_menu.addAction(self.tr("Append All"))
        self.folk54_SSC_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.folk54_SSC_diagram, self._dataset, True))
        self.BP12_GSM_menu = self.menu.addMenu(self.tr("Plot GSM Diagram (Blott and Pye, 2012)"))
        self.BP12_GSM_plot_selected_action = self.BP12_GSM_menu.addAction(self.tr("Plot"))
        self.BP12_GSM_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.BP12_GSM_diagram, self.selections, False))
        self.BP12_GSM_append_selected_action = self.BP12_GSM_menu.addAction(self.tr("Append"))
        self.BP12_GSM_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.BP12_GSM_diagram, self.selections, True))
        self.BP12_GSM_plot_all_action = self.BP12_GSM_menu.addAction(self.tr("Plot All"))
        self.BP12_GSM_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.BP12_GSM_diagram, self._dataset, False))
        self.BP12_GSM_append_all_action = self.BP12_GSM_menu.addAction(self.tr("Append All"))
        self.BP12_GSM_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.BP12_GSM_diagram, self._dataset, True))
        self.BP12_SSC_menu = self.menu.addMenu(self.tr("Plot SSC Diagram (Blott and Pye, 2012)"))
        self.BP12_SSC_plot_selected_action = self.BP12_SSC_menu.addAction(self.tr("Plot"))
        self.BP12_SSC_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.BP12_SSC_diagram, self.selections, False))
        self.BP12_SSC_append_selected_action = self.BP12_SSC_menu.addAction(self.tr("Append"))
        self.BP12_SSC_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.BP12_SSC_diagram, self.selections, True))
        self.BP12_SSC_plot_all_action = self.BP12_SSC_menu.addAction(self.tr("Plot All"))
        self.BP12_SSC_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.BP12_SSC_diagram, self._dataset, False))
        self.BP12_SSC_append_all_action = self.BP12_SSC_menu.addAction(self.tr("Append All"))
        self.BP12_SSC_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.BP12_SSC_diagram, self._dataset, True))
        self.CM_menu = self.menu.addMenu(self.tr("Plot C-M Diagram"))
        self.CM_plot_selected_action = self.CM_menu.addAction(self.tr("Plot"))
        self.CM_plot_selected_action.triggered.connect(
            lambda: self.plot_chart(self.CM_diagram, self.selections, False))
        self.CM_append_selected_action = self.CM_menu.addAction(self.tr("Append"))
        self.CM_append_selected_action.triggered.connect(
            lambda: self.plot_chart(self.CM_diagram, self.selections, True))
        self.CM_plot_all_action = self.CM_menu.addAction(self.tr("Plot All"))
        self.CM_plot_all_action.triggered.connect(
            lambda: self.plot_chart(self.CM_diagram, self._dataset, False))
        self.CM_append_all_action = self.CM_menu.addAction(self.tr("Append All"))
        self.CM_append_all_action.triggered.connect(
            lambda: self.plot_chart(self.CM_diagram, self._dataset, True))
        self.previous_button.setEnabled(False)
        self.current_page_combo_box.setEnabled(False)
        self.next_button.setEnabled(False)
        self.data_table.customContextMenuRequested.connect(self.show_menu)
        self.cumulative_chart = CumulativeChart()
        self.frequency_chart = FrequencyChart()
        self.frequency_3D_chart = Frequency3DChart()
        self.frequency_heatmap = FrequencyHeatmap()
        self.folk54_GSM_diagram = Folk54GSMDiagramChart()
        self.folk54_SSC_diagram = Folk54SSCDiagramChart()
        self.BP12_GSM_diagram = BP12GSMDiagramChart()
        self.BP12_SSC_diagram = BP12SSCDiagramChart()
        self.CM_diagram = CMDiagramChart()
        self.normal_msg = QtWidgets.QMessageBox(self)

    def show_menu(self, pos):
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

    def on_dataset_loaded(self, dataset: Dataset):
        if dataset is None:
            return
        self._dataset = dataset
        self.current_page_combo_box.clear()
        page_count, left = divmod(len(self._dataset), self.PAGE_ROWS)
        if left != 0:
            page_count += 1
        self.current_page_combo_box.addItems([self.tr("Page {0}").format(i + 1) for i in range(page_count)])
        self.previous_button.setEnabled(True)
        self.current_page_combo_box.setEnabled(True)
        self.next_button.setEnabled(True)

        self.update_page(0)

    @property
    def tips(self) -> List[str]:
        tips = [
            self.tr("By clicking the option at menu bar, you can load the grain size distributions \
                (Menu -> Open -> Grain Size Dataset)."),
            self.tr("By clicking the option at menu bar, you can save the statistical parameters and \
                classification groups to a Excel file (Menu -> Save -> Statistical Result)."),
            self.tr("By right clicking at the table region, you can open the menu to draw charts.")]
        return tips

    @property
    def is_geometric(self) -> bool:
        return self.geometric_checkbox.isChecked()

    def on_is_geometric_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.geometric_checkbox.setText(self.tr("Geometric [unit is {0}]").format(self.unit))
        else:
            self.geometric_checkbox.setText(self.tr("Logarithmic [unit is {0}]").format(self.unit))
        self.update_page(self.page_index)

    @property
    def is_FW57(self) -> bool:
        return self.FW57_checkbox.isChecked()

    def on_is_FW57_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.FW57_checkbox.setText(self.tr("Method of Folk and Ward (1957)"))
        else:
            self.FW57_checkbox.setText(self.tr("Method of Statistical Moments"))
        self.update_page(self.page_index)

    @property
    def supported_proportions(self) -> Sequence[Tuple[str, str]]:
        result = (
            ("proportions_gsm", self.tr("Gravel, Sand, Mud")),
            ("proportions_ssc", self.tr("Sand, Silt, Clay")),
            ("proportions_bgssc", self.tr("Boulder, Gravel, Sand, Silt, Clay")))
        return result

    @property
    def proportion(self) -> Tuple[str, str]:
        index = self.proportion_combo_box.currentIndex()
        key, description = self.supported_proportions[index]
        return key, description

    @property
    def page_index(self) -> int:
        return self.current_page_combo_box.currentIndex()

    @property
    def n_pages(self) -> int:
        return self.current_page_combo_box.count()

    @property
    def unit(self) -> str:
        return self.tr("micron") if self.is_geometric else self.tr("phi")

    def update_page(self, page_index: int):
        if self._dataset is None:
            return

        def write(row: int, col: int, value: Union[int, float, str]):
            if isinstance(value, str):
                pass
            elif isinstance(value, int):
                value = str(value)
            elif isinstance(value, float):
                value = f"{value: 0.2f}"
            else:
                value = value.__str__()
            item = QtWidgets.QTableWidgetItem(value)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.data_table.setItem(row, col, item)

        # necessary to clear
        self.data_table.clear()
        if page_index == self.n_pages - 1:
            start = page_index * self.PAGE_ROWS
            end = len(self._dataset)
        else:
            start, end = page_index * self.PAGE_ROWS, (page_index + 1) * self.PAGE_ROWS
        proportion_key, proportion_name = self.proportion
        col_names = [self.tr("Mean [{0}]").format(self.unit),
                     self.tr("Mean Description"),
                     self.tr("Median [{0}]").format(self.unit),
                     self.tr("Modes [{0}]").format(self.unit),
                     self.tr("Sorting Coefficient"),
                     self.tr("Sorting Description"),
                     self.tr("Skewness"),
                     self.tr("Skewness Description"),
                     self.tr("Kurtosis"),
                     self.tr("Kurtosis Description"),
                     self.tr("({0}) Proportion [%]").format(proportion_name),
                     self.tr("Group (Folk, 1954)"),
                     self.tr("Group Symbol (Blott and Pye, 2012)"),
                     self.tr("Group (Blott and Pye, 2012)")]
        col_keys = [(True, "mean"),
                    (True, "mean_description"),
                    (True, "median"),
                    (True, "modes"),
                    (True, "std"),
                    (True, "std_description"),
                    (True, "skewness"),
                    (True, "skewness_description"),
                    (True, "kurtosis"),
                    (True, "kurtosis_description"),
                    (False, proportion_key),
                    (False, "group_folk54"),
                    (False, "group_bp12_symbol"),
                    (False, "group_bp12")]
        self.data_table.setRowCount(end - start)
        self.data_table.setColumnCount(len(col_names))
        self.data_table.setHorizontalHeaderLabels(col_names)
        self.data_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap)
        self.data_table.setVerticalHeaderLabels(self._dataset.sample_names[start: end])
        for row, sample in enumerate(self._dataset[start: end]):
            statistics = all_statistics(sample.classes, sample.classes_phi, sample.distribution)
            if self.is_geometric:
                if self.is_FW57:
                    sub_key = "geometric_fw57"
                else:
                    sub_key = "geometric"
            else:
                if self.is_FW57:
                    sub_key = "logarithmic_fw57"
                else:
                    sub_key = "logarithmic"
            for col, (in_sub, key) in enumerate(col_keys):
                value = statistics[sub_key][key] if in_sub else statistics[key]
                if key == "modes":
                    write(row, col, ", ".join([f"{m:0.2f}" for m in value]))
                elif key[:11] == "proportions":
                    write(row, col, ", ".join([f"{p * 100:0.2f}" for p in value]))
                else:
                    write(row, col, value)

        self.data_table.resizeColumnsToContents()

    @property
    def selections(self):
        if self._dataset is None:
            return []
        start = self.page_index * self.PAGE_ROWS
        temp = set()
        for item in self.data_table.selectedRanges():
            for i in range(item.topRow(), min(self.PAGE_ROWS + 1, item.bottomRow() + 1)):
                temp.add(i + start)
        indexes = list(temp)
        indexes.sort()
        samples = [self._dataset[i] for i in indexes]
        return samples

    def on_previous_button_clicked(self):
        if self.page_index > 0:
            self.current_page_combo_box.setCurrentIndex(self.page_index - 1)

    def on_next_button_clicked(self):
        if self.page_index < self.n_pages - 1:
            self.current_page_combo_box.setCurrentIndex(self.page_index + 1)

    def plot_chart(self, chart, samples: Iterable[Sample], append: bool):
        copy_samples = list(samples)
        if self._dataset is None:
            self.show_error(self.tr("The dataset has not been loaded."))
        elif len(copy_samples) == 0:
            self.show_error(self.tr("No sample was selected."))
        else:
            if isinstance(chart, DiagramChart):
                kwargs = {"mfc": highlight_color()}
            else:
                kwargs = {}
            chart.show_samples(copy_samples, append=append, **kwargs)
            chart.show()

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.data_table.setHorizontalHeaderLabels([self.tr("Tips")])
        if self._dataset is None:
            for row, tip in enumerate(self.tips):
                item = QtWidgets.QTableWidgetItem(tip)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.data_table.setItem(row, 0, item)
            self.current_page_combo_box.setItemText(0, self.tr("No Page"))
        else:
            self.update_page(self.page_index)
            for i in range(self.n_pages):
                self.current_page_combo_box.setItemText(i, self.tr("Page {0}").format(i + 1))
        self.previous_button.setText(self.tr("Previous"))
        self.previous_button.setToolTip(self.tr("Click to back to the previous page."))
        self.next_button.setText(self.tr("Next"))
        self.next_button.setToolTip(self.tr("Click to jump to the next page."))
        if self.is_geometric:
            self.geometric_checkbox.setText(self.tr("Geometric [unit is {0}]").format(self.unit))
        else:
            self.geometric_checkbox.setText(self.tr("Logarithmic [unit is {0}]").format(self.unit))
        if self.is_FW57:
            self.FW57_checkbox.setText(self.tr("Method of Folk and Ward (1957)"))
        else:
            self.FW57_checkbox.setText(self.tr("Method of Statistical Moments"))
        for i, (_, description) in enumerate(self.supported_proportions):
            self.proportion_combo_box.setItemText(i, description)
        self.plot_cumulative_menu.setTitle(self.tr("Plot Cumulative Frequency Chart"))
        self.cumulative_plot_selected_action.setText(self.tr("Plot"))
        self.cumulative_append_selected_action.setText(self.tr("Append"))
        self.cumulative_plot_all_action.setText(self.tr("Plot All"))
        self.cumulative_append_all_action.setText(self.tr("Append All"))
        self.plot_frequency_menu.setTitle(self.tr("Plot Frequency Distribution Chart"))
        self.frequency_plot_selected_action.setText(self.tr("Plot"))
        self.frequency_append_selected_action.setText(self.tr("Append"))
        self.frequency_plot_all_action.setText(self.tr("Plot All"))
        self.frequency_append_all_action.setText(self.tr("Append All"))
        self.plot_frequency_3D_menu.setTitle(self.tr("Plot Frequency 3D Chart"))
        self.frequency_3D_plot_selected_action.setText(self.tr("Plot"))
        self.frequency_3D_append_selected_action.setText(self.tr("Append"))
        self.frequency_3D_plot_all_action.setText(self.tr("Plot All"))
        self.frequency_3D_append_all_action.setText(self.tr("Append All"))
        self.plot_frequency_heatmap_menu.setTitle(self.tr("Plot Frequency Heatmap"))
        self.frequency_heatmap_plot_selected_action.setText(self.tr("Plot"))
        self.frequency_heatmap_append_selected_action.setText(self.tr("Append"))
        self.frequency_heatmap_plot_all_action.setText(self.tr("Plot All"))
        self.frequency_heatmap_append_all_action.setText(self.tr("Append All"))
        self.folk54_GSM_menu.setTitle(self.tr("Plot GSM Diagram (Folk, 1954)"))
        self.folk54_GSM_plot_selected_action.setText(self.tr("Plot"))
        self.folk54_GSM_append_selected_action.setText(self.tr("Append"))
        self.folk54_GSM_plot_all_action.setText(self.tr("Plot All"))
        self.folk54_GSM_append_all_action.setText(self.tr("Append All"))
        self.folk54_SSC_menu.setTitle(self.tr("Plot SSC Diagram (Folk, 1954)"))
        self.folk54_SSC_plot_selected_action.setText(self.tr("Plot"))
        self.folk54_SSC_append_selected_action.setText(self.tr("Append"))
        self.folk54_SSC_plot_all_action.setText(self.tr("Plot All"))
        self.folk54_SSC_append_all_action.setText(self.tr("Append All"))
        self.BP12_GSM_menu.setTitle(self.tr("Plot GSM Diagram (Blott and Pye, 2012)"))
        self.BP12_GSM_plot_selected_action.setText(self.tr("Plot"))
        self.BP12_GSM_append_selected_action.setText(self.tr("Append"))
        self.BP12_GSM_plot_all_action.setText(self.tr("Plot All"))
        self.BP12_GSM_append_all_action.setText(self.tr("Append All"))
        self.BP12_SSC_menu.setTitle(self.tr("Plot SSC Diagram (Blott and Pye, 2012)"))
        self.BP12_SSC_plot_selected_action.setText(self.tr("Plot"))
        self.BP12_SSC_append_selected_action.setText(self.tr("Append"))
        self.BP12_SSC_plot_all_action.setText(self.tr("Plot All"))
        self.BP12_SSC_append_all_action.setText(self.tr("Append All"))
        self.CM_menu.setTitle(self.tr("Plot C-M Diagram"))
        self.CM_plot_selected_action.setText(self.tr("Plot"))
        self.CM_append_selected_action.setText(self.tr("Append"))
        self.CM_plot_all_action.setText(self.tr("Plot All"))
        self.CM_append_all_action.setText(self.tr("Append All"))
