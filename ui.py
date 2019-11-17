import pyqtgraph as pg
from PyQt5.QtWidgets import  QWidget, QGridLayout, QSizePolicy
from PyQt5.QtCore import Qt, QMutex

import numpy as np


class FittingCanvas(QWidget):

    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.init_ui()
        self.__mutex = QMutex()

    def init_ui(self):
        self.setGeometry(300,300,300,200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout = QGridLayout(self)
        self.plot_widget = pg.PlotWidget(title="Fitting Windows")
        self.main_layout.addWidget(self.plot_widget)
        
        self.__target_style = dict(pen=None, symbol="o", symbolBrush=pg.mkBrush("y"), symbolSize=6)
        self.__target_data_item = pg.PlotDataItem(name="Target", **self.__target_style)
        self.plot_widget.addItem(self.__target_data_item)
        
        self.__fitted_style = dict(pen=pg.mkPen("k", width=3, style=Qt.DashLine))
        self.__fitted_data_item = pg.PlotDataItem(name="Fit", **self.__fitted_style)
        self.plot_widget.addItem(self.__fitted_data_item)
        
        self.__fitted_component_styles = [
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
        self.__fitted_component_data_items=[]
        self.plot_widget.plotItem.setLogMode(x=True)

    
    def on_ncomp_changed(self, ncomp: int):
        # Check the validity of `ncomp`
        if type(ncomp) != int:
            raise TypeError(ncomp)
        if ncomp <= 0:
            raise ValueError(ncomp)

        self.__mutex.lock()
        # clear
        for data in self.__fitted_component_data_items:
            self.plot_widget.plotItem.removeItem(data)
        self.__fitted_component_data_items.clear()
        # add data items
        for i in range(ncomp):
            data = pg.PlotDataItem(name="C{0}".format(i+1), **self.__fitted_component_styles[i])
            self.plot_widget.addItem(data)
            self.__fitted_component_data_items.append(data)
        self.__mutex.unlock()


    def on_epoch_finished(self, data):
        self.__mutex.lock()
        self.__target_data_item.setData(*data[0], **self.__target_style)
        self.__fitted_data_item.setData(*data[1], **self.__fitted_style)

        for (x, y), data_item, style in zip(data[2:-1], self.__fitted_component_data_items, self.__fitted_component_styles):
            data_item.setData(x, y, **style)

        self.__mutex.unlock()

    def on_single_iteration_finished(self, data):
        isLocked = self.__mutex.tryLock()
        if not isLocked:
            return
        self.__target_data_item.setData(*data[0], **self.__target_style)
        self.__fitted_data_item.setData(*data[1], **self.__fitted_style)

        for (x, y), data_item, style in zip(data[2:-1], self.__fitted_component_data_items, self.__fitted_component_styles):
            data_item.setData(x, y, **style)
        self.__mutex.unlock()
