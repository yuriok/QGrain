import math

import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QSizePolicy, QWidget

from data import FittedData

pg.setConfigOptions(foreground=pg.mkColor("k"), background=pg.mkColor("w"), antialias=True, )

class FittingCanvas(QWidget):
    sigWidgetsEnable = pyqtSignal(bool)

    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.current_iteration = 0
        self.sample_id = "Unknown"
        self.init_ui()

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout = QGridLayout(self)
        self.plot_widget = pg.PlotWidget(
            title="Fitting Windows", enableMenu=True)
        self.main_layout.addWidget(self.plot_widget)

        self.target_style = dict(pen=None, symbol="o", symbolBrush=pg.mkBrush(
            "y"), symbolPen=None, symbolSize=6)
        self.target_item = pg.PlotDataItem(name="Target", **self.target_style)
        self.plot_widget.plotItem.addItem(self.target_item)

        self.sum_style = dict(pen=pg.mkPen("w", width=3, style=Qt.DashLine))
        self.sum_item = pg.PlotDataItem(name="Fitted", **self.sum_style)
        self.plot_widget.plotItem.addItem(self.sum_item)
        self.plot_widget.plotItem.setLabels(
            left="Probability Density", bottom="Grain size (μm)")
        self.plot_widget.plotItem.showGrid(True, True)
        # self.plot_widget.plotItem.setMenuEnabled(False)
        
        self.component_styles = [
            dict(pen=pg.mkPen("#FF5600", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#0D58A6", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#53DF00", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#BF6030", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#A63800", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#689CD2", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#CD0074", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#FFFF00", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#640CAB", width=2, style=Qt.DashLine)),
            dict(pen=pg.mkPen("#FFCB00", width=2, style=Qt.DashLine))]
        self.component_curves = []
        self.component_lines = []
        self.plot_widget.plotItem.setLogMode(x=True)

        self.legend = pg.LegendItem(offset=(50, 50))
        self.legend.setParentItem(self.plot_widget.plotItem)
        self.legend.addItem(self.target_item, "Target")
        self.legend.addItem(self.sum_item, "Fitted")

    def on_ncomp_changed(self, ncomp: int):
        # Check the validity of `ncomp`
        if type(ncomp) != int:
            raise TypeError(ncomp)
        if ncomp <= 0:
            raise ValueError(ncomp)

        # clear
        for name, curve in self.component_curves:
            self.plot_widget.plotItem.removeItem(curve)
            self.legend.removeItem(name)
        for line in self.component_lines:
            self.plot_widget.plotItem.removeItem(line)
        self.component_curves.clear()
        self.component_lines.clear()
        # add
        for i in range(ncomp):
            component_name = "C{0}".format(i+1)
            curve = pg.PlotDataItem(name=component_name,
                                   **self.component_styles[i])
            line = pg.InfiniteLine(angle=90, movable=False, pen=self.component_styles[i]["pen"])
            self.plot_widget.plotItem.addItem(curve)
            self.plot_widget.plotItem.addItem(line)
            self.legend.addItem(curve, component_name)
            self.component_curves.append((component_name, curve))
            self.component_lines.append(line)

    def on_target_data_changed(self, sample_id, x, y):
        self.sample_id = sample_id
        self.plot_widget.plotItem.setTitle(sample_id)
        # self.target_item.setData(x, y, **self.target_style)

    def on_epoch_finished(self, data: FittedData):
        self.target_item.setData(*data.target, **self.target_style)
        self.sum_item.setData(*data.sum, **self.sum_style)

        for (x, y), (name, curve_item), style in zip(data.components, self.component_curves, self.component_styles):
            curve_item.setData(x, y, **style)
        for i, line_item in enumerate(self.component_lines):
            line_item.setValue(math.log10(data.statistic[i]["mean"]))
        self.sigWidgetsEnable.emit(True)
        self.current_iteration = 0

    def on_single_iteration_finished(self, data: FittedData):
        # Iteration will take too much times, so disable the ui to reject additional requests
        # UI will be enable at `on_epoch_finished` 
        self.sigWidgetsEnable.emit(False)
        self.target_item.setData(*data.target, **self.target_style)
        self.sum_item.setData(*data.sum, **self.sum_style)
        for (x, y), (name, curve_item), style in zip(data.components, self.component_curves, self.component_styles):
            curve_item.setData(x, y, **style)
        for i, line_item in enumerate(self.component_lines):
            line_item.setValue(math.log10(data.statistic[i]["mean"]))
        self.plot_widget.plotItem.setTitle("{0} iter({1})".format(self.sample_id, self.current_iteration))
        self.current_iteration += 1
