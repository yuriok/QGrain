__all__ = ["ParameterEditor"]

from typing import *

import numpy as np
from PySide6 import QtCore, QtWidgets
from numpy import ndarray

from ..statistics import interval_phi
from ..models import DistributionType, SSUResult, ArtificialDataset
from ..distributions import sort_components
from ..charts.DistributionChart import DistributionChart


class ParameterComponent(QtWidgets.QWidget):
    NORMAL_SETTINGS = dict(
        n_parameters=3,
        parameter_names=("Location", "Scale", "Weight"),
        ranges=((-15.0, 15.0), (0.01, 100.0), (0.0, 10.0)),
        defaults=(5.0, 1.0, 1.0),
        steps=(0.1, 0.1, 0.1))

    SKEW_NORMAL_SETTINGS = dict(
        n_parameters=4,
        parameter_names=("Shape", "Location", "Scale", "Weight"),
        ranges=((-100.0, 100.0), (-15.0, 15.0), (0.01, 100.0), (0.0, 10.0)),
        defaults=(0.0, 5.0, 1.0, 1.0),
        steps=(0.1, 0.1, 0.1, 0.1))

    WEIBULL_SETTINGS = dict(
        n_parameters=3,
        parameter_names=("Shape", "Scale", "Weight"),
        ranges=((-2000.0, 2000.0), (0.01, 2000.0), (0.0, 10.0)),
        defaults=(3.6, 1.0, 1.0),
        steps=(0.1, 0.1, 0.1))

    GENERAL_WEIBULL_SETTINGS = dict(
        n_parameters=4,
        parameter_names=("Shape", "Location", "Scale", "Weight"),
        ranges=((-2000.0, 2000.0), (-2000.0, 2000.0), (0.01, 2000.0), (0.0, 10.0)),
        defaults=(3.6, 5.0, 1.0, 1.0),
        steps=(0.1, 0.1, 0.1, 0.1))

    SETTING_MAP = {
        DistributionType.Normal: NORMAL_SETTINGS,
        DistributionType.SkewNormal: SKEW_NORMAL_SETTINGS,
        DistributionType.Weibull: WEIBULL_SETTINGS,
        DistributionType.GeneralWeibull: GENERAL_WEIBULL_SETTINGS}

    value_changed = QtCore.Signal()

    def __init__(self, name: str, distribution_type: DistributionType, parent=None):
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._name = name
        self._distribution_type = distribution_type
        # make sure that the pyside2-lupdate.exe can recognize these keywords
        _ = [self.tr("Shape"), self.tr("Location"), self.tr("Scale"), self.tr("Weight")]
        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.group = QtWidgets.QGroupBox("")
        self.group.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.main_layout.addWidget(self.group)
        self.group_layout = QtWidgets.QGridLayout(self.group)
        self.group_layout.setColumnMinimumWidth(0, 32)
        self.group_layout.setColumnMinimumWidth(1, 32)
        settings = self.SETTING_MAP[self._distribution_type]
        self.widgets = [] # type: list[tuple[QtWidgets.QLabel, QtWidgets.QDoubleSpinBox]]
        self.name_label = QtWidgets.QLabel(self._name)
        self.name_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.value_label = QtWidgets.QLabel(self.tr("Value"))
        self.value_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.group_layout.addWidget(self.name_label, 0, 0)
        self.group_layout.addWidget(self.value_label, 0, 1)
        for i in range(settings["n_parameters"]):
            label = QtWidgets.QLabel(self.tr(settings["parameter_names"][i]))
            label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            input = QtWidgets.QDoubleSpinBox()
            input.setRange(*settings["ranges"][i])
            input.setSingleStep(settings["steps"][i])
            input.setValue(settings["defaults"][i])
            self.group_layout.addWidget(label, i+1, 0)
            self.group_layout.addWidget(input, i+1, 1)
            self.widgets.append((label, input))
        for _, input in self.widgets:
            input.valueChanged.connect(self.on_value_changed)

    @property
    def parameters(self):
        return [input.value() for _, input in self.widgets]

    def set_parameters(self, parameters):
        self.blockSignals(True)
        for (_, input), value in zip(self.widgets, parameters):
            input.setValue(value)
        self.blockSignals(False)

    def on_value_changed(self, _):
        self.value_changed.emit()

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.value_label.setText(self.tr("Value"))
        settings = self.SETTING_MAP[self._distribution_type]
        for param_name, (param_label, _) in zip(settings["parameter_names"], self.widgets):
            param_label.setText(self.tr(param_name))


class ParameterEditor(QtWidgets.QDialog):
    SUPPORT_DISTRIBUTIONS = (
        (DistributionType.Normal, "Normal"),
        (DistributionType.SkewNormal, "Skew Normal"),
        (DistributionType.Weibull, "Weibull"),
        (DistributionType.GeneralWeibull, "General Weibull"))
    DISTRIBUTION_INDEX_MAP = {
        DistributionType.Normal: 0,
        DistributionType.SkewNormal: 1,
        DistributionType.Weibull: 2,
        DistributionType.GeneralWeibull: 3}

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._cache_dict = {}
        self._classes = np.logspace(0, 5, 101) * 0.02
        self._classes_phi = -np.log2(self._classes / 1000.0)
        self._interval_phi = interval_phi(self._classes_phi)
        self._target = None
        self.setWindowTitle(self.tr("Parameter Editor"))
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.control_group = QtWidgets.QGroupBox(self.tr("Control"))
        self.control_layout = QtWidgets.QGridLayout(self.control_group)
        self.preset_label = QtWidgets.QLabel(self.tr("Preset Archive"))
        self.preset_combo_box = QtWidgets.QComboBox()
        self.preset_combo_box.addItems([str(i) for i in range(1, 11)])
        self.preset_combo_box.currentIndexChanged.connect(self.switch_preset)
        self.enabled_checkbox = QtWidgets.QCheckBox()
        self.enabled_checkbox.setText(self.tr("Enabled"))
        self.preview_button = QtWidgets.QPushButton(self.tr("Preview"))
        self.preview_button.clicked.connect(self.update_chart)
        self.control_layout.addWidget(self.preset_label, 0, 0)
        self.control_layout.addWidget(self.preset_combo_box, 0, 1)
        self.control_layout.addWidget(self.enabled_checkbox, 0, 2)
        self.control_layout.addWidget(self.preview_button, 0, 3)
        self.n_components_label = QtWidgets.QLabel(self.tr("Number of Components"))
        self.n_components_input = QtWidgets.QSpinBox()
        self.n_components_input.setRange(1, 12)
        self.n_components_input.valueChanged.connect(self.on_n_components_changed)
        self.n_components_input.valueChanged.connect(self.update_cache)
        self.distribution_type_label = QtWidgets.QLabel(self.tr("Distribution Type"))
        self.distribution_type_combo_box = QtWidgets.QComboBox()
        self.distribution_type_combo_box.addItems([name for _, name in self.SUPPORT_DISTRIBUTIONS])
        self.distribution_type_combo_box.currentIndexChanged.connect(self.on_distribution_type_changed)
        self.distribution_type_combo_box.currentIndexChanged.connect(self.on_distribution_type_changed)
        self.control_layout.addWidget(self.n_components_label, 1, 0)
        self.control_layout.addWidget(self.n_components_input, 1, 1)
        self.control_layout.addWidget(self.distribution_type_label, 1, 2)
        self.control_layout.addWidget(self.distribution_type_combo_box, 1, 3)
        self.parameter_tab_widget = QtWidgets.QTabWidget()
        self.component_sets: List[Tuple[QtWidgets.QWidget, QtWidgets.QGridLayout, List[ParameterComponent]]] = []
        self.preview_group = QtWidgets.QGroupBox(self.tr("Preview"))
        self.preview_layout = QtWidgets.QGridLayout(self.preview_group)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_chart = DistributionChart(parent=self)
        self.preview_layout.addWidget(self.preview_chart, 0, 0)
        self.splitter_1 = QtWidgets.QSplitter()
        self.splitter_1.setOrientation(QtCore.Qt.Vertical)
        self.splitter_1.addWidget(self.control_group)
        self.splitter_1.addWidget(self.parameter_tab_widget)
        self.splitter_1.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.splitter_2 = QtWidgets.QSplitter()
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.addWidget(self.splitter_1)
        self.splitter_2.addWidget(self.preview_group)
        self.main_layout.addWidget(self.splitter_2, 0, 0)

        self.switch_preset(0)
        self.file_dialog = QtWidgets.QFileDialog(parent=self)

    @property
    def preset_index(self) -> int:
        return self.preset_combo_box.currentIndex()

    @property
    def n_components(self) -> int:
        return self.n_components_input.value()

    @property
    def distribution_type(self) -> DistributionType:
        return self.SUPPORT_DISTRIBUTIONS[self.distribution_type_combo_box.currentIndex()][0]

    @property
    def parameter_enabled(self) -> bool:
        return self.enabled_checkbox.isChecked()

    @property
    def parameters(self) -> np.ndarray:
        parameters = []
        for _, _, components in self.component_sets:
            for component in components:
                if component.isEnabled():
                    parameters.append(component.parameters)
        parameters = np.expand_dims(np.array(parameters).T, axis=0)
        classes = np.expand_dims(np.expand_dims(self._classes_phi, axis=0), axis=0).repeat(self.n_components, axis=1)
        sorted_parameters = sort_components(self.distribution_type, parameters)
        return sorted_parameters[0]

    def _clear_components(self):
        for param_holder, param_layout, components in self.component_sets:
            for component in components:
                component.value_changed.disconnect(self.update_cache)
                param_layout.removeWidget(component)
                component.hide()
            param_holder.hide()
        self.parameter_tab_widget.clear()
        self.component_sets.clear()

    def _add_components(self):
        for i in range(4):
            param_holder = QtWidgets.QWidget()
            self.parameter_tab_widget.addTab(param_holder, f"C{i * 3 + 1}-{i * 3 + 3}")
            param_layout = QtWidgets.QGridLayout(param_holder)
            param_layout.setContentsMargins(0, 0, 0, 0)
            components = []
            for j in range(3):
                component = ParameterComponent(f"C{i*3+j+1}", self.distribution_type)
                component.value_changed.connect(self.update_cache)
                param_layout.addWidget(component, 0, j)
                components.append(component)
            self.component_sets.append((param_holder, param_layout, components))

    def on_n_components_changed(self, n_components: int):
        for i in range(4):
            for j in range(3):
                current_index = i*3+j
                if current_index < n_components:
                    self.component_sets[i][2][j].setEnabled(True)
                else:
                    self.component_sets[i][2][j].setEnabled(False)

    def on_distribution_type_changed(self, _):
        self._clear_components()
        self._add_components()
        self.on_n_components_changed(self.n_components)

    def update_cache(self):
        self._cache_dict[self.preset_index] = (self.distribution_type, self.n_components, self.parameters)

    def switch_preset(self, preset_index: int):
        if preset_index in self._cache_dict.keys():
            distribution_type, n_components, parameter_matrix = self._cache_dict[preset_index]
            self.distribution_type_combo_box.setCurrentIndex(self.DISTRIBUTION_INDEX_MAP[distribution_type])
            self.n_components_input.setValue(n_components)
            parameters = parameter_matrix.T
        else:
            self.distribution_type_combo_box.setCurrentIndex(3)
            self.n_components_input.setValue(3)
            parameters = [
                [611.76, -467.20, 477.40, 0.81],
                [2.28, 5.35, 2.57, 2.12],
                [2.74, 2.67, 2.68, 2.77]]
        enabled_components = [] # type: list[ParameterComponent]
        for _, _, components in self.component_sets:
            for component in components:
                if component.isEnabled():
                    enabled_components.append(component)
        for component_parameters, component in zip(parameters, enabled_components):
            component.set_parameters(component_parameters)
        self.update_cache()
        self.update_chart()

    def setup_target(self, classes: np.ndarray, target: np.ndarray):
        self._classes = classes
        self._classes_phi = -np.log2(classes / 1000.0)
        self._interval_phi = interval_phi(self._classes_phi)
        self._target = target

    def update_chart(self):
        sample = ArtificialDataset(np.expand_dims(self.parameters, axis=0), self.distribution_type)[0]
        self.preview_chart.show_chart(sample)

    def refer_parameters(self, distribution_type: DistributionType, parameters: ndarray):
        assert isinstance(distribution_type, DistributionType)
        assert isinstance(parameters, ndarray)
        assert parameters.ndim == 2
        n_parameters, n_components = parameters.shape
        self.distribution_type_combo_box.setCurrentIndex(self.DISTRIBUTION_INDEX_MAP[distribution_type])
        self.n_components_input.setValue(n_components)
        enabled_components: List[ParameterComponent] = []
        for _, _, components in self.component_sets:
            for component in components:
                if component.isEnabled():
                    enabled_components.append(component)
        for i, component in enumerate(enabled_components):
            component.set_parameters(parameters[:, i])
        self.update_chart()
        self.update_cache()

    def refer_result(self, result: SSUResult):
        self.distribution_type_combo_box.setCurrentIndex(self.DISTRIBUTION_INDEX_MAP[result.distribution_type])
        self.n_components_input.setValue(len(result))
        enabled_components: List[ParameterComponent] = []
        for _, _, components in self.component_sets:
            for component in components:
                if component.isEnabled():
                    enabled_components.append(component)
        for i, component in enumerate(enabled_components):
            component.set_parameters(result.parameters[-1, :, i])
        self.update_chart()
        self.update_cache()

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("Parameter Editor"))
        self.control_group.setTitle(self.tr("Control"))
        self.preset_label.setText(self.tr("Preset Archive"))
        self.enabled_checkbox.setText(self.tr("Enabled"))
        self.preview_button.setText(self.tr("Preview"))
        self.n_components_label.setText(self.tr("Number of Components"))
        self.distribution_type_label.setText(self.tr("Distribution Type"))
        self.preview_group.setTitle(self.tr("Preview"))
