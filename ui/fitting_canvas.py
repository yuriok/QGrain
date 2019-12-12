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
        self.sample_id = self.tr("Unknown")
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
        self.plot_widget.plotItem.setLabel("bottom", self.tr("Grain size (μm)"), **self.label_styles)
        self.title_format = """<font face="Times New Roman">%s</font>"""
        self.plot_widget.plotItem.setTitle(self.title_format % "Fitting Canvas")
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
        self.logger.debug("Received the component changed signal, start to change clear and add data items.")
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

    def on_target_data_changed(self, sample_id, x, y):
        self.sample_id = sample_id
        self.plot_widget.plotItem.setTitle(self.title_format % sample_id)
        self.logger.debug("Target data has been changed to [%s].", sample_id)

    def on_fitting_epoch_suceeded(self, data: FittedData):
        non_nan_data = data.get_non_nan_copy()
        self.target_item.setData(*non_nan_data.target, **self.target_style)
        self.sum_item.setData(*non_nan_data.sum, **self.sum_style)

        for (x, y), (name, curve_item), style in zip(non_nan_data.components, self.component_curves, self.component_styles):
            curve_item.setData(x, y, **style)
        for i, line_item in enumerate(self.component_lines):
            line_item.setValue(math.log10(non_nan_data.statistic[i]["mean"]))
        self.logger.debug("Epoch fitting finished. Data of DataItem has updated.")


    def on_single_iteration_finished(self, current_iteration, data: FittedData):
        non_nan_data = data.get_non_nan_copy()
        self.target_item.setData(*non_nan_data.target, **self.target_style)
        self.sum_item.setData(*non_nan_data.sum, **self.sum_style)
        for (x, y), (name, curve_item), style in zip(non_nan_data.components, self.component_curves, self.component_styles):
            curve_item.setData(x, y, **style)
        for i, line_item in enumerate(self.component_lines):
            line_item.setValue(math.log10(non_nan_data.statistic[i]["mean"]))
        self.plot_widget.plotItem.setTitle(self.title_format % "{0} iter({1})".format(self.sample_id, current_iteration))

