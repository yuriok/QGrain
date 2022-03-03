import logging
import typing

from PySide6 import QtCore, QtWidgets

from ..artificial import (SIMPLE_PRESET, DistributionType,
                          get_dataset, get_mean_sample, get_sample)
from ..chart.DistributionChart import DistributionChart
from ..io import save_artificial_dataset


class GeneratorComponent(QtWidgets.QWidget):
    NORMAL_SETTINGS = dict(
        n_params = 3,
        param_names=("Location", "Scale", "Weight"),
        mean_ranges=((-15.0, 15.0), (0.01, 100.0), (0.0, 10.0)),
        mean_defaults=(5.0, 1.0, 1.0),
        mean_steps=(0.1, 0.1, 0.1),
        std_ranges=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)),
        std_defaults=(0.0, 0.0, 0.1),
        std_steps=(0.1, 0.1, 0.1))

    SKEW_NORMAL_SETTINGS = dict(
        n_params = 4,
        param_names=("Shape", "Location", "Scale", "Weight"),
        mean_ranges=((-100.0, 100.0), (-15.0, 15.0), (0.01, 100.0), (0.0, 10.0)),
        mean_defaults=(0.0, 5.0, 1.0, 1.0),
        mean_steps=(0.1, 0.1, 0.1, 0.1),
        std_ranges=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)),
        std_defaults=(0.0, 0.0, 0.0, 0.1),
        std_steps=(0.1, 0.1, 0.1, 0.1))

    WEIBULL_SETTINGS = dict(
        n_params = 3,
        param_names=("Shape", "Scale", "Weight"),
        mean_ranges=((-500.0, 500.0), (0.01, 500.0), (0.0, 10.0)),
        mean_defaults=(3.6, 1.0, 1.0),
        mean_steps=(0.1, 0.1, 0.1),
        std_ranges=((0.0, 100.0), (0.0, 100.0), (0.0, 10.0)),
        std_defaults=(0.0, 0.0, 0.1),
        std_steps=(0.1, 0.1, 0.1))

    GENERAL_WEIBULL_SETTINGS = dict(
        n_params = 4,
        param_names=("Shape", "Location", "Scale", "Weight"),
        mean_ranges=((-500.0, 500.0), (-500.0, 500.0), (0.01, 500.0), (0.0, 10.0)),
        mean_defaults=(3.6, 5.0, 1.0, 1.0),
        mean_steps=(0.1, 0.1, 0.1, 0.1),
        std_ranges=((0.0, 100.0), (0.0, 100.0), (0.0, 100.0), (0.0, 10.0)),
        std_defaults=(0.0, 0.0, 0.0, 0.1),
        std_steps=(0.1, 0.1, 0.1, 0.1))

    SETTING_MAP = {
        DistributionType.Normal: NORMAL_SETTINGS,
        DistributionType.SkewNormal: SKEW_NORMAL_SETTINGS,
        DistributionType.Weibull: WEIBULL_SETTINGS,
        DistributionType.GeneralWeibull: GENERAL_WEIBULL_SETTINGS}

    value_changed = QtCore.Signal()
    def __init__(self, name: str, distribution_type: DistributionType, parent=None):
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.__name = name
        self.__distribution_type = distribution_type
        self.init_ui()

    def init_ui(self):
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
        self.group_layout.setColumnMinimumWidth(2, 32)
        settings = self.SETTING_MAP[self.__distribution_type]
        self.widgets = [] # type: list[tuple[QtWidgets.QLabel, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox]]
        self.name_label = QtWidgets.QLabel(self.__name)
        self.name_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.mean_label = QtWidgets.QLabel(self.tr("Mean"))
        self.mean_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.std_label = QtWidgets.QLabel(self.tr("Standard\nDeviation"))
        self.std_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.group_layout.addWidget(self.name_label, 0, 0)
        self.group_layout.addWidget(self.mean_label, 0, 1)
        self.group_layout.addWidget(self.std_label, 0, 2)
        for i in range(settings["n_params"]):
            label = QtWidgets.QLabel(self.tr(settings["param_names"][i]))
            label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            mean_input = QtWidgets.QDoubleSpinBox()
            mean_input.setRange(*settings["mean_ranges"][i])
            mean_input.setSingleStep(settings["mean_steps"][i])
            mean_input.setValue(settings["mean_defaults"][i])
            std_input = QtWidgets.QDoubleSpinBox()
            std_input.setRange(*settings["std_ranges"][i])
            std_input.setSingleStep(settings["std_steps"][i])
            std_input.setValue(settings["std_defaults"][i])
            self.group_layout.addWidget(label, i+1, 0)
            self.group_layout.addWidget(mean_input, i+1, 1)
            self.group_layout.addWidget(std_input, i+1, 2)
            self.widgets.append((label, mean_input, std_input))

        for _, mean_input, std_input in self.widgets:
            mean_input.valueChanged.connect(self.on_value_changed)
            std_input.valueChanged.connect(self.on_value_changed)

    @property
    def target(self):
        target = [(mean_input.value(), std_input.value()) for _, mean_input, std_input in self.widgets]
        return target

    @target.setter
    def target(self, values: list):
        for (mean, std), (_, mean_input, std_input) in zip(values, self.widgets):
            mean_input.setValue(mean)
            std_input.setValue(std)

    def on_value_changed(self, _):
        self.value_changed.emit()

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.mean_label.setText(self.tr("Mean"))
        self.std_label.setText(self.tr("Standard\nDeviation"))
        settings = self.SETTING_MAP[self.__distribution_type]
        for param_name, (param_label, _, _) in zip(settings["param_names"], self.widgets):
            param_label.setText(self.tr(param_name))


class DatasetGenerator(QtWidgets.QWidget):
    logger = logging.getLogger("QGrain")

    SUPPORT_DISTRIBUTIONS = (
        DistributionType.Normal,
        DistributionType.SkewNormal,
        DistributionType.Weibull,
        DistributionType.GeneralWeibull)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_ui()
        self.distribution_type_combo_box.setCurrentIndex(1)
        self.target = SIMPLE_PRESET["target"]
        self.minimum_size_input.setValue(0.02)
        self.maximum_size_input.setValue(2000.0)
        self.n_classes_input.setValue(101)
        self.precision_input.setValue(4)
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        self.normal_msg = QtWidgets.QMessageBox(self)
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(lambda: self.update_chart(True))

    def init_ui(self):
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.control_group = QtWidgets.QGroupBox(self.tr("Control"))
        self.control_layout = QtWidgets.QGridLayout(self.control_group)
        self.control_layout.setColumnStretch(0, 1)
        self.control_layout.setColumnStretch(1, 1)
        self.n_components_label = QtWidgets.QLabel(self.tr("Number of Components"))
        self.n_components_input = QtWidgets.QSpinBox()
        self.n_components_input.setRange(1, 12)
        self.n_components_input.valueChanged.connect(self.on_n_components_changed)
        self.distribution_type_label = QtWidgets.QLabel(self.tr("Distribution Type"))
        self.distribution_type_combo_box = QtWidgets.QComboBox()
        self.distribution_type_combo_box.addItems([distribution_type.value for distribution_type in self.SUPPORT_DISTRIBUTIONS])
        self.distribution_type_combo_box.currentIndexChanged.connect(self.on_distribution_type_changed)
        self.control_layout.addWidget(self.n_components_label, 0, 0)
        self.control_layout.addWidget(self.n_components_input, 0, 1)
        self.control_layout.addWidget(self.distribution_type_label, 0, 2)
        self.control_layout.addWidget(self.distribution_type_combo_box, 0, 3)

        self.minimum_size_label = QtWidgets.QLabel(self.tr("Minimum Size [{0}]").format("μm"))
        self.minimum_size_input = QtWidgets.QDoubleSpinBox()
        self.minimum_size_input.setDecimals(2)
        self.minimum_size_input.setRange(1e-4, 1e6)
        self.minimum_size_input.setValue(0.0200)
        self.maximum_size_label = QtWidgets.QLabel(self.tr("Maximum Size [{0}]").format("μm"))
        self.maximum_size_input = QtWidgets.QDoubleSpinBox()
        self.maximum_size_input.setDecimals(2)
        self.maximum_size_input.setRange(1e-4, 1e6)
        self.maximum_size_input.setValue(2000.0000)
        self.control_layout.addWidget(self.minimum_size_label, 1, 0)
        self.control_layout.addWidget(self.minimum_size_input, 1, 1)
        self.control_layout.addWidget(self.maximum_size_label, 1, 2)
        self.control_layout.addWidget(self.maximum_size_input, 1, 3)

        self.n_classes_label = QtWidgets.QLabel(self.tr("Number of Classes"))
        self.n_classes_input = QtWidgets.QSpinBox()
        self.n_classes_input.setRange(10, 1e4)
        self.n_classes_input.setValue(101)
        self.precision_label = QtWidgets.QLabel(self.tr("Data Precision"))
        self.precision_input = QtWidgets.QSpinBox()
        self.precision_input.setRange(2, 8)
        self.precision_input.setValue(4)
        self.control_layout.addWidget(self.n_classes_label, 2, 0)
        self.control_layout.addWidget(self.n_classes_input, 2, 1)
        self.control_layout.addWidget(self.precision_label, 2, 2)
        self.control_layout.addWidget(self.precision_input, 2, 3)

        self.n_samples_label = QtWidgets.QLabel(self.tr("Number of Samples"))
        self.n_samples_input = QtWidgets.QSpinBox()
        self.n_samples_input.setRange(100, 100000)
        self.control_layout.addWidget(self.n_samples_label, 3, 0, 1, 2)
        self.control_layout.addWidget(self.n_samples_input, 3, 2, 1, 2)

        self.preview_button = QtWidgets.QPushButton(self.tr("Preview"))
        self.preview_button.clicked.connect(self.on_preview_clicked)
        self.generate_button = QtWidgets.QPushButton(self.tr("Generate"))
        self.generate_button.clicked.connect(self.on_generate_clicked)
        self.control_layout.addWidget(self.preview_button, 4, 0, 1, 2)
        self.control_layout.addWidget(self.generate_button, 4, 2, 1, 2)

        self.param_tab_widget = QtWidgets.QTabWidget()
        self.component_sets = [] # type: list[tuple[QtWidgets.QWidget, QtWidgets.QGridLayout, list[GeneratorComponent]]]
        self._add_components()

        self.preview_group = QtWidgets.QGroupBox(self.tr("Preview"))
        self.preview_layout = QtWidgets.QGridLayout(self.preview_group)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.chart = DistributionChart(parent=self, show_mode=True)
        self.preview_layout.addWidget(self.chart, 0, 0)

        self.splitter_1 = QtWidgets.QSplitter()
        self.splitter_1.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_1.addWidget(self.control_group)
        self.splitter_1.addWidget(self.preview_group)
        # self.splitter_1.setStretchFactor(0, 2)
        # self.splitter_1.setStretchFactor(1, 1)
        self.splitter_2 = QtWidgets.QSplitter()
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.addWidget(self.splitter_1)
        self.splitter_2.addWidget(self.param_tab_widget)
        self.main_layout.addWidget(self.splitter_2, 0, 0)

    @property
    def n_components(self) -> int:
        return self.n_components_input.value()

    @property
    def distribution_type(self) -> DistributionType:
        index = self.distribution_type_combo_box.currentIndex()
        distribution_type = self.SUPPORT_DISTRIBUTIONS[index]
        return distribution_type

    @property
    def components(self) -> typing.List[GeneratorComponent]:
        result = []
        for _, _, components in self.component_sets:
            for component in components:
                if component.isEnabled():
                    result.append(component)
        return result

    @property
    def target(self):
        return [component.target for component in self.components]

    @target.setter
    def target(self, values):
        if len(values) != len(self.components):
            self.n_components_input.setValue(len(values))
        for component_widget, component_target in zip(self.components, values):
            component_widget.blockSignals(True)
            component_widget.target = component_target
            component_widget.blockSignals(False)
        self.update_chart()

    @property
    def func_kwargs(self):
        min_μm = self.minimum_size_input.value()
        max_μm = self.maximum_size_input.value()
        n_classes = self.n_classes_input.value()
        if min_μm > max_μm:
            min_μm, max_μm = max_μm, min_μm
        precision = self.precision_input.value()
        noise = precision + 1
        kwargs = dict(
            target=self.target,
            distribution_type=self.distribution_type,
            min_μm=min_μm,
            max_μm=max_μm,
            n_classes=n_classes,
            precision=precision,
            noise=noise)
        return kwargs

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

    def _clear_components(self):
        for i, (param_holder, param_layout, components) in enumerate(self.component_sets):
            for j, component in enumerate(components):
                component.value_changed.disconnect(self.on_value_changed)
                param_layout.removeWidget(component)
                component.hide()
            param_holder.hide()
        self.param_tab_widget.clear()
        self.component_sets.clear()

    def _add_components(self):
        for i in range(4):
            param_holder = QtWidgets.QWidget()
            self.param_tab_widget.addTab(param_holder, f"AC{i*3+1}-{i*3+3}")
            param_layout = QtWidgets.QGridLayout(param_holder)
            param_layout.setContentsMargins(0, 0, 0, 0)
            components = []
            for j in range(3):
                component = GeneratorComponent(f"AC{i*3+j+1}", self.distribution_type)
                component.value_changed.connect(self.on_value_changed)
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

    def on_preview_clicked(self):
        if self.update_timer.isActive():
            self.preview_button.setText(self.tr("Preview"))
            self.update_timer.stop()
            self.update_chart()
        else:
            self.preview_button.setText(self.tr("Stop"))
            self.update_timer.start(500)

    def on_generate_clicked(self):
        if self.update_timer.isActive():
            self.preview_button.setText(self.tr("Preview"))
            self.update_timer.stop()
            self.update_chart()
        self.generate_button.setEnabled(False)
        filename, _ = self.file_dialog.getSaveFileName(
            self, self.tr("Choose a filename to save the generated dataset"),
            None, "Microsoft Excel (*.xlsx)")
        if filename is None or filename == "":
            return
        n_samples = self.n_samples_input.value()
        dataset = self.get_random_dataset(n_samples)
        save_artificial_dataset(dataset, filename)
        self.generate_button.setEnabled(True)
        self.logger.info(f"Generated dataset has been saved to the Excel file: [{filename}].")

    def get_random_sample(self):
        if self.minimum_size_input.value() == self.maximum_size_input.value():
            return
        sample = get_sample(**self.func_kwargs)
        sample.name = self.tr("Artificial Sample")
        return sample

    def get_mean_sample(self):
        if self.minimum_size_input.value() == self.maximum_size_input.value():
            return
        sample = get_mean_sample(**self.func_kwargs)
        sample.name = self.tr("Artificial Sample")
        return sample

    def get_random_dataset(self, n_samples: int):
        if self.minimum_size_input.value() == self.maximum_size_input.value():
            return
        dataset = get_dataset(**self.func_kwargs, n_samples=n_samples)
        return dataset

    def on_value_changed(self):
        self.update_chart()

    def update_chart(self, random=False):
        if not random:
            sample = self.get_mean_sample()
        else:
            sample = self.get_random_sample()
        quick = self.chart.last_model is not None and self.chart.last_model.n_components == sample.n_components
        self.chart.show_model(sample.view_model, quick)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.control_group.setTitle(self.tr("Control"))
        self.preview_group.setTitle(self.tr("Preview"))
        self.n_components_label.setText(self.tr("Number of Components"))
        self.distribution_type_label.setText(self.tr("Distribution Type"))
        self.minimum_size_label.setText(self.tr("Minimum Size [{0}]").format("μm"))
        self.maximum_size_label.setText(self.tr("Maximum Size [{0}]").format("μm"))
        self.n_classes_label.setText(self.tr("Number of Classes"))
        self.precision_label.setText(self.tr("Data Precision"))
        self.n_samples_label.setText(self.tr("Number of Samples"))
        if self.update_timer.isActive():
            self.preview_button.setText(self.tr("Stop"))
        else:
            self.preview_button.setText(self.tr("Preview"))
        self.generate_button.setText(self.tr("Generate"))
