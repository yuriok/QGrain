import logging

import numpy as np
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QMutex, QPointF, Qt, Signal
from PySide2.QtGui import QBrush, QColor, QFont, QPen

from models.FittingResult import FittingResult
from models.SampleData import SampleData
from ui.Canvas import Canvas
from ui.InfiniteLine import InfiniteLine

from typing import List

class DistributionCanvas(Canvas):
    sigExpectedMeanValueChanged = Signal(tuple)
    logger = logging.getLogger("root.ui.DistributionCanvas")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, isDark=True):
        super().__init__(parent)
        self.initChart()
        self.setupChartStyle()
        self.setThemeMode(isDark)
        self.chart.legend().detachFromChart()
        self.chart.legend().setPos(100.0, 60.0)
        self.__infiniteLineMutex = QMutex()

    def initChart(self):
        # init axes
        self.axisX = QtCharts.QLogValueAxis()
        self.axisX.setBase(10.0)
        self.axisX.setMinorTickCount(-1)
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.axisY = QtCharts.QValueAxis()
        self.chart.addAxis(self.axisY, Qt.AlignLeft)

        # init two const series
        self.targetSeries = QtCharts.QScatterSeries()
        self.targetSeries.setName(self.tr("Target"))
        self.targetSeries.setMarkerSize(5.0)
        self.targetSeries.setPen(QPen(QColor(255, 255, 255, 0)))
        self.chart.addSeries(self.targetSeries)
        self.fittedSeries = QtCharts.QLineSeries()
        self.fittedSeries.setName(self.tr("Fitted"))
        self.chart.addSeries(self.fittedSeries)
        # attach series to axes
        self.targetSeries.attachAxis(self.axisX)
        self.targetSeries.attachAxis(self.axisY)
        self.fittedSeries.attachAxis(self.axisX)
        self.fittedSeries.attachAxis(self.axisY)

        # set title
        self.chart.setTitle(self.tr("Distribution Canvas"))
        # set labels
        self.axisX.setTitleText(self.tr("Grain size")+" (μm)")
        self.axisY.setTitleText(self.tr("Probability Density"))

        self.componentSeries = [] # type: List[QtChart.QLineSeries]
        self.componentInfiniteLines = [] # type: List[InfiniteLine]

    def on_component_number_changed(self, component_number: int):
        self.logger.info("Received the component changed signal, start to clear and add data items.")
        # Check the validity of `component_number`
        if type(component_number) != int:
            raise TypeError(component_number)
        if component_number <= 0:
            raise ValueError(component_number)
        # clear
        for series in self.componentSeries:
            self.chart.removeSeries(series)
        for line in self.componentInfiniteLines:
            line.disconnectFromChart()
        self.componentSeries.clear()
        self.componentInfiniteLines.clear()
        self.logger.debug("Items cleared.")
        # add
        for i in range(component_number):
            componentName = "C{0}".format(i+1)
            # series
            series = QtCharts.QLineSeries()
            series.setName(componentName)
            self.chart.addSeries(series)
            series.attachAxis(self.axisX)
            series.attachAxis(self.axisY)
            # line
            line = InfiniteLine(isHorizontal=False, chart=self.chart,
                                pen=series.pen(), callback=self.on_infinite_line_moved)
            self.componentSeries.append(series)
            self.componentInfiniteLines.append(line)
        # update the size of legend
        self.chart.legend().setMinimumSize(150.0, 30*(2+component_number))
        self.logger.debug("Items added.")

    def on_target_data_changed(self, sample: SampleData):
        # update the title of canvas
        if sample.name is None or sample.name == "":
            sample.name = "UNKNOWN"
        self.chart.setTitle(sample.name)
        self.targetSeries.replace(self.toPoints(sample.classes, sample.distribution))
        self.fittedSeries.clear()
        for series in self.componentSeries:
            series.clear()
        for line in self.componentInfiniteLines:
            line.value = 1.0
        self.axisX.setRange(sample.classes[0], sample.classes[-1])
        self.axisY.setRange(0.0, np.max(sample.distribution)*1.2)
        self.logger.debug("Target data has been changed to [%s].", sample.name)

    def update_canvas_by_data(self, result: FittingResult, current_iteration=None):
        # update the title of canvas
        if current_iteration is not None:
            self.chart.setTitle(("{0} "+self.tr("Iteration")+" ({1})").format(
                result.name, current_iteration))
        # update fitted series
        self.fittedSeries.replace(self.toPoints(result.real_x, result.fitted_y))
        # update component series
        for i, (component, series, line) in enumerate(zip(
                result.components,
                self.componentSeries,
                self.componentInfiniteLines)):
            series.replace(self.toPoints(result.real_x, component.component_y))
            # display the median (modal size) and fraction for users
            componentName = "C{0}".format(i+1)
            series.setName(componentName + " ({0:.1f} μm, {1:.1%})".format(component.median, component.fraction))
            if np.isnan(component.mean) or np.isinf(component.mean):
                # if mean is invalid, set the position of this line to initial position
                line.value = 1.0
            else:
                line.value = component.mean

    def on_fitting_epoch_suceeded(self, result: FittingResult):
        self.update_canvas_by_data(result)
        self.exportToPng("./temp/distribution_canvas/png/{0} - {1} - {2}.png".format(
            result.name, result.distribution_type, result.component_number))
        self.exportToSvg("./temp/distribution_canvas/svg/{0} - {1} - {2}.svg".format(
            result.name, result.distribution_type, result.component_number))

    def on_single_iteration_finished(self, current_iteration: int, result: FittingResult):
        self.update_canvas_by_data(result, current_iteration=current_iteration)

    def on_infinite_line_moved(self):
        # this method will be called by another object directly
        # use lock to keep safety
        self.__infiniteLineMutex.lock()
        expectedMeanValues = tuple(sorted([line.value for line in self.componentInfiniteLines]))
        self.sigExpectedMeanValueChanged.emit(expectedMeanValues)
        self.__infiniteLineMutex.unlock()
