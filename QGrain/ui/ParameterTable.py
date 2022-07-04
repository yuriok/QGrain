import logging
import typing

from PySide6 import QtCore, QtWidgets

from ..ssu import DistributionType, SSUResult


class ParameterTable(QtWidgets.QDialog):
    parameter_name_map = {
        DistributionType.Normal: ("Location", "Scale", "Weight"),
        DistributionType.SkewNormal: ("Shape", "Location", "Scale", "Weight"),
        DistributionType.Weibull: ("Shape", "Scale", "Weight"),
        DistributionType.GeneralWeibull: ("Shape", "Location", "Scale", "Weight")
    }

    def __init__(self, result: SSUResult, parent=None):
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.init_ui(result)

    def init_ui(self, result: SSUResult):
        # make sure that the pyside2-lupdate.exe can recognize these keywords
        _ = [self.tr("Shape"), self.tr("Location"), self.tr("Scale"), self.tr("Weight")]
        self.main_layout = QtWidgets.QGridLayout(self)
        self.setMinimumSize(200, 160)
        self.setWindowTitle(self.tr("Resolved Parameters"))
        parameter_names = self.parameter_name_map[result.distribution_type]
        self.headers = [] # type: list[tuple[str, QtWidgets.QLabel]]
        for i, name in enumerate(parameter_names):
            header = QtWidgets.QLabel(self.tr(name))
            self.main_layout.addWidget(header, 0, i+1)
            self.headers.append((name, header))
        for i in range(result.n_components):
            row = i+1
            component_name = QtWidgets.QLabel(f"C{i+1}")
            self.main_layout.addWidget(component_name, row, 0)
            for j in range(len(parameter_names)):
                col = j+1
                parameter = QtWidgets.QLabel(f"{result.components[i].parameters[j]:.4f}")
                self.main_layout.addWidget(parameter, row, col)

    def changeEvent(self, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.LanguageChange:
            self.retranslate()

    def retranslate(self):
        self.setWindowTitle(self.tr("Resolved Parameters"))
        for name, header in self.headers:
            header.setText(self.tr(name))
