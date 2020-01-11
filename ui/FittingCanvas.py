import logging
from enum import Enum, unique

import numpy as np
import pyqtgraph as pg
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QGridLayout, QWidget

from models.FittingResult import FittingResult
from models.SampleData import SampleData

pg.setConfigOptions(foreground=pg.mkColor("k"), background=pg.mkColor("#FFFFFF00"), antialias=True)

@unique
class XAxisSpace(Enum):
    Raw = 0
    Log10 = 1
    Phi = 2

class FittingCanvas(QWidget):
    sigExpectedMeanValueChanged = Signal(tuple)
    logger = logging.getLogger("root.ui.FittingCanvas")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget(enableMenu=True)
        self.main_layout.addWidget(self.plot_widget)
        self.target_style = dict(pen=None, symbol="o", symbolBrush=pg.mkBrush("#161B26"), symbolPen=None, symbolSize=5)
        self.target_item = pg.PlotDataItem(name="Target", **self.target_style)
        self.plot_widget.plotItem.addItem(self.target_item)
        self.sum_style = dict(pen=pg.mkPen("#062170", width=3, style=Qt.DashLine))
        self.fitted_item = pg.PlotDataItem(name="Fitted", **self.sum_style)
        self.plot_widget.plotItem.addItem(self.fitted_item)
        self.label_styles = {"font-family": "Times New Roman"}
        self.plot_widget.plotItem.setLabel("left", self.tr("Probability Density"), **self.label_styles)
        self.plot_widget.plotItem.setLabel("bottom", self.tr("Grain size")+" [μm]", **self.label_styles)
        self.title_format = """<font face="Times New Roman">%s</font>"""
        self.plot_widget.plotItem.setTitle(self.title_format % self.tr("Fitting Canvas"))
        self.plot_widget.plotItem.showGrid(True, True)
        self.tickFont = QFont("Arial")
        self.tickFont.setPointSize(8)
        self.plot_widget.plotItem.getAxis("left").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("bottom").tickFont = self.tickFont
        # self.plot_widget.plotItem.setMenuEnabled(False)
        self.legend_format = """<font face="Times New Roman">%s</font>"""
        self.legend = pg.LegendItem(offset=(80, 50))
        self.legend.setParentItem(self.plot_widget.plotItem)
        self.legend.addItem(self.target_item, self.legend_format % self.tr("Target"))
        self.legend.addItem(self.fitted_item, self.legend_format % self.tr("Fitted"))
        self.x_axis_space = XAxisSpace.Log10
        self.component_curves = []
        self.component_lines = []
        self.position_limit = None
        self.position_cache = []
        self.component_styles = [
            dict(pen=pg.mkPen("#600CAC", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#0E51A7", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#FFC900", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#EA0037", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#C0F400", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#00AA72", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#53DF00", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#FF7100", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#FD0006", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#009B95", width=2, style=Qt.DashLine))]


    def raw2space(self, value: float) -> float:
        processed = value
        if self.position_limit is not None:
            lower, upper = self.position_limit
            if processed < lower:
                processed = lower
            elif processed > upper:
                processed = upper
        if self.x_axis_space == XAxisSpace.Raw:
            return processed
        elif self.x_axis_space == XAxisSpace.Log10:
            return np.log10(processed)
        elif self.x_axis_space == XAxisSpace.Phi:
            return -np.log2(processed)
        else:
            raise NotImplementedError(self.x_axis_space)

    def space2raw(self, value: float) -> float:
        if self.x_axis_space == XAxisSpace.Raw:
            processed = value
        elif self.x_axis_space == XAxisSpace.Log10:
            processed = 10**value
        elif self.x_axis_space == XAxisSpace.Phi:
            processed = 2**(-value)
        else:
            raise NotImplementedError(self.x_axis_space)
        if self.position_limit is not None:
            lower, upper = self.position_limit
            if processed < lower:
                processed = lower
            elif processed > upper:
                processed = upper
        return processed

    def on_line_position_changed(self, current_line):
        for i, line in enumerate(self.component_lines):
            if current_line is line:
                raw_value = self.space2raw(current_line.getXPos())
                current_line.setValue(self.raw2space(raw_value))
                self.position_cache[i] = raw_value
        self.sigExpectedMeanValueChanged.emit(tuple(self.position_cache))

    def on_component_number_changed(self, ncomp: int):
        self.logger.info("Received the component changed signal, start to change clear and add data items.")
        # Check the validity of `ncomp`
        if type(ncomp) != int:
            raise TypeError(ncomp)
        if ncomp <= 0:
            raise ValueError(ncomp)
        # clear
        for name, curve in self.component_curves:
            self.plot_widget.plotItem.removeItem(curve)
            self.legend.removeItem(curve)
        for line in self.component_lines:
            self.plot_widget.plotItem.removeItem(line)
        self.component_curves.clear()
        self.component_lines.clear()
        self.position_cache = np.ones(ncomp)
        self.logger.debug("Items cleared.")
        # add
        for i in range(ncomp):
            component_name = "C{0}".format(i+1)
            curve = pg.PlotDataItem(name=component_name,**self.component_styles[i])
            line = pg.InfiniteLine(angle=90, movable=False, pen=self.component_styles[i]["pen"])
            line.setMovable(True)
            line.sigDragged.connect(self.on_line_position_changed)
            self.plot_widget.plotItem.addItem(curve)
            self.plot_widget.plotItem.addItem(line)
            self.legend.addItem(curve, self.legend_format % component_name)
            self.component_curves.append((component_name, curve))
            self.component_lines.append(line)
        self.logger.debug("Items added.")

    def on_target_data_changed(self, sample: SampleData):
        # change the value space of x axis
        if self.x_axis_space == XAxisSpace.Raw:
            self.plot_widget.plotItem.setLogMode(x=False)
        elif self.x_axis_space == XAxisSpace.Log10:
            self.plot_widget.plotItem.setLogMode(x=True)
        else:
            raise NotImplementedError(self.x_axis_space)

        # the range to limit the positions of lines
        self.position_limit = (sample.classes[0], sample.classes[-1])
        x_axis = self.plot_widget.plotItem.getAxis("bottom")
        x_axis.enableAutoSIPrefix(enable=False)
        major_ticks= [(self.raw2space(x_value), "{0:0.2f}".format(x_value)) for i, x_value in enumerate(sample.classes) if i%20==0]
        minor_ticks= [(self.raw2space(x_value), "{0:0.2f}".format(x_value)) for i, x_value in enumerate(sample.classes) if i%5==0]
        all_ticks = [(self.raw2space(x_value), "{0:0.2f}".format(x_value)) for i, x_value in enumerate(sample.classes)]
        x_axis.setTicks([major_ticks, minor_ticks, all_ticks])
        self.logger.debug("Target data has been changed to [%s].", sample.name)

        # update the title of canvas
        if sample.name is None or sample.name == "":
            sample.name = "UNKNOWN"
        self.plot_widget.plotItem.setTitle(self.title_format % sample.name)
        # update target
        # target data (i.e. grain size classes and distribution) should have no nan value indeed
        # it should be checked during load data progress
        self.target_item.setData(sample.classes, sample.distribution, **self.target_style)
        self.fitted_item.clear()
        for name, curve in self.component_curves:
            curve.clear()
        for line in self.component_lines:
            line.setValue(1)

    def update_canvas_by_data(self, result: FittingResult, current_iteration=None):
        # TODO: check component number
        # update the title of canvas
        sample_name = result.name
        if sample_name is None or sample_name == "":
            sample_name = "UNKNOWN"
        if current_iteration is None:
            self.plot_widget.plotItem.setTitle(self.title_format % sample_name)
        else:
            self.plot_widget.plotItem.setTitle(self.title_format % "{0} iter({1})".format(sample_name, current_iteration))
        # update fitted
        self.fitted_item.setData(result.real_x, result.fitted_y, **self.sum_style)
        # update component curves
        for i, component, (name, curve), style, line in zip(range(result.component_number),
                                                           result.components, self.component_curves,
                                                           self.component_styles, self.component_lines):
            curve.setData(result.real_x, component.component_y, **style)
            if np.isnan(component.mean) or np.isinf(component.mean):
                continue
            space_value = self.raw2space(component.mean)
            self.position_cache[i] = self.space2raw(space_value)
            line.setValue(space_value)

    def on_fitting_epoch_suceeded(self, result: FittingResult):
        self.update_canvas_by_data(result)

    def on_single_iteration_finished(self, current_iteration, result: FittingResult):
        self.update_canvas_by_data(result, current_iteration=current_iteration)
