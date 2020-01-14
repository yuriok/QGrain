import logging

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PySide2.QtCore import Qt
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QGridLayout, QWidget

from models.FittingResult import FittingResult


class LossCanvas(QWidget):
    logger = logging.getLogger("root.ui.FittingCanvas")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget(enableMenu=False)
        self.main_layout.addWidget(self.plot_widget, 0, 0)
        # add image exporters
        self.png_exporter = ImageExporter(self.plot_widget.plotItem)
        self.svg_exporter = SVGExporter(self.plot_widget.plotItem)
        # show all axis
        self.plot_widget.plotItem.showAxis("left")
        self.plot_widget.plotItem.showAxis("right")
        self.plot_widget.plotItem.showAxis("top")
        self.plot_widget.plotItem.showAxis("bottom")
        # plot data item
        self.style = dict(pen=pg.mkPen("#062170", width=3))
        self.plot_data_item = pg.PlotDataItem(name="Loss", **self.style)
        self.plot_widget.plotItem.addItem(self.plot_data_item)
        # set labels
        self.label_styles = {"font-family": "Times New Roman"}
        self.plot_widget.plotItem.setLabel("left", self.tr("Loss"), **self.label_styles)
        self.plot_widget.plotItem.setLabel("bottom", self.tr("Iteration"), **self.label_styles)
        # set title
        self.title_format = """<font face="Times New Roman">%s</font>"""
        self.plot_widget.plotItem.setTitle(self.title_format % self.tr("Loss Canvas"))
        # show grids
        self.plot_widget.plotItem.showGrid(True, True)
        # set the font of ticks
        self.tickFont = QFont("Arial")
        self.tickFont.setPointSize(8)
        self.plot_widget.plotItem.getAxis("left").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("right").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("top").tickFont = self.tickFont
        self.plot_widget.plotItem.getAxis("bottom").tickFont = self.tickFont
        # set legend
        self.legend_format = """<font face="Times New Roman">%s</font>"""
        self.legend = pg.LegendItem(offset=(80, 50))
        self.legend.setParentItem(self.plot_widget.plotItem)
        self.legend.addItem(self.plot_data_item, self.legend_format % self.tr("Loss"))
        # set y log
        self.plot_widget.plotItem.setLogMode(y=True)

        # data
        self.x = []
        self.y = []

    def on_single_iteration_finished(self, current_iteration: int, result: FittingResult):
        if current_iteration == 0:
            if len(self.x) != 0 and len(self.y) != 0:
                self.x.clear()
                self.y.clear()
                # save figures
                self.png_exporter.export("./temp/current_loss_canvas.png")
                self.svg_exporter.export("./temp/current_loss_canvas.svg")
            else:
                self.plot_widget.plotItem.setTitle(self.title_format % result.name)
        loss = result.mean_squared_error
        self.x.append(current_iteration)
        self.y.append(loss)
        self.plot_data_item.setData(self.x, self.y)
