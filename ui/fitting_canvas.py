import logging
import math

import numpy as np
import pyqtgraph as pg
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QGridLayout, QSizePolicy, QWidget

from data import FittedData

pg.setConfigOptions(foreground=pg.mkColor("k"), background=pg.mkColor("#FFFFFF00"), antialias=True)

class FittingCanvas(QWidget):
    logger = logging.getLogger("root.ui.FittingCanvas")
    gui_logger = logging.getLogger("GUI")
    
    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.init_ui()

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout = QGridLayout(self)
        self.plot_widget = pg.PlotWidget(enableMenu=True)
        self.main_layout.addWidget(self.plot_widget)
        self.target_style = dict(pen=None, symbol="o", symbolBrush=pg.mkBrush("#161B26"), symbolPen=None, symbolSize=5)
        self.target_item = pg.PlotDataItem(name="Target", **self.target_style)
        self.plot_widget.plotItem.addItem(self.target_item)
        
        self.sum_style = dict(pen=pg.mkPen("#062170", width=3, style=Qt.DashLine))
        self.sum_item = pg.PlotDataItem(name="Fitted", **self.sum_style)
        self.plot_widget.plotItem.addItem(self.sum_item)
        self.label_styles = {"font-family": "Times New Roman"}
        self.plot_widget.plotItem.setLabel("left", self.tr("Probability Density"), **self.label_styles)
        self.plot_widget.plotItem.setLabel("bottom", self.tr("Grain size"), **self.label_styles)
        self.title_format = """<font face="Times New Roman">%s</font>"""
        self.plot_widget.plotItem.setTitle(self.title_format % self.tr("Fitting Canvas"))
        self.plot_widget.plotItem.showGrid(True, True)
        self.tickFont = QFont("Arial")
        self.tickFont.setPointSize(8)
        self.plot_widget.plotItem.getAxis("left").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("bottom").tickFont = self.tickFont
        # self.plot_widget.plotItem.setMenuEnabled(False)
        self.plot_widget.plotItem.setLogMode(x=True)
        self.legend_format = """<font face="Times New Roman">%s</font>"""
        self.legend = pg.LegendItem(offset=(80, 50))
        self.legend.setParentItem(self.plot_widget.plotItem)
        self.legend.addItem(self.target_item, self.legend_format % self.tr("Target"))
        self.legend.addItem(self.sum_item, self.legend_format % self.tr("Fitted Sum"))
        self.component_curves = []
        self.component_lines = []
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


    # TODO: add lock because if ncomp changed during iteration, the data items will also change
    def on_ncomp_changed(self, ncomp: int):
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
        self.logger.debug("Items cleared.")
        # add
        for i in range(ncomp):
            component_name = "C{0}".format(i+1)
            curve = pg.PlotDataItem(name=component_name,**self.component_styles[i])
            line = pg.InfiniteLine(angle=90, movable=False, pen=self.component_styles[i]["pen"])
            self.plot_widget.plotItem.addItem(curve)
            self.plot_widget.plotItem.addItem(line)
            self.legend.addItem(curve, self.legend_format % component_name)
            self.component_curves.append((component_name, curve))
            self.component_lines.append(line)
        self.logger.debug("Items added.")

    def on_target_data_changed(self, sample_name, x, y):
        self.logger.debug("Target data has been changed to [%s].", sample_name)

    def update_canvas_by_data(self, data: FittedData, current_iteration=None):
        # update the title of canvas
        sample_name = data.name
        if sample_name is None or sample_name =="":
            sample_name = "UNKNOWN"
        if current_iteration is None:
            self.plot_widget.plotItem.setTitle(self.title_format % sample_name)
        else:
            self.plot_widget.plotItem.setTitle(self.title_format % "{0} iter({1})".format(sample_name, current_iteration))
        # update target
        # target data (i.e. grain size classes and distribution) should have no nan value indeed
        # it should be checked during load data progress
        self.target_item.setData(*data.target, **self.target_style)
        # update sum
        sum_x, sum_y = data.sum
        self.sum_item.setData(sum_x, np.nan_to_num(sum_y), **self.sum_style)
        # update components
        for (x, y), (name, curve_item), style in zip(data.components, self.component_curves, self.component_styles):
            curve_item.setData(x, np.nan_to_num(y), **style)
        for i, line_item in enumerate(self.component_lines):
            mean_value = data.statistic[i]["mean"]
            # jump this iteration if value is nan or inf
            if np.isnan(mean_value) or np.isinf(mean_value):
                continue
            # because x axis is in log mode, it's necessary to calculate the log10 to make it correct
            line_item.setValue(math.log10(mean_value))

    def on_fitting_epoch_suceeded(self, data: FittedData):
        self.update_canvas_by_data(data)

    def on_single_iteration_finished(self, current_iteration, data: FittedData):
        self.update_canvas_by_data(data, current_iteration=current_iteration)
