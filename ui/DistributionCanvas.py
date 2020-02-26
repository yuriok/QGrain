import logging
from typing import List

import numpy as np
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QMutex, Qt, Signal
from PySide2.QtGui import QColor, QPen

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())
from models.FittingResult import FittingResult
from models.SampleData import SampleData
from ui.Canvas import Canvas
from ui.InfiniteLine import InfiniteLine


class DistributionCanvas(Canvas):
    sigExpectedMeanValueChanged = Signal(tuple)
    logger = logging.getLogger("root.ui.DistributionCanvas")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, is_dark=True):
        super().__init__(parent)
        self.set_theme_mode(is_dark)
        self.init_chart()
        self.setup_chart_style()
        self.chart.legend().detachFromChart()
        self.chart.legend().setPos(100.0, 60.0)
        self.__infinite_line_mutex = QMutex()
        # used to check if it's necessary to update the component series and lines
        self.__current_component_number = None

    def init_chart(self):
        # init axes
        self.axis_x = QtCharts.QLogValueAxis()
        self.axis_x.setBase(10.0)
        self.axis_x.setMinorTickCount(-1)
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.axis_y = QtCharts.QValueAxis()
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)

        # init two const series
        self.target_series = QtCharts.QScatterSeries()
        self.target_series.setName(self.tr("Target"))
        self.target_series.setMarkerSize(5.0)
        self.target_series.setPen(QPen(QColor(255, 255, 255, 0)))
        self.chart.addSeries(self.target_series)
        self.fitted_series = QtCharts.QLineSeries()
        self.fitted_series.setName(self.tr("Fitted"))
        self.chart.addSeries(self.fitted_series)
        # attach series to axes
        self.target_series.attachAxis(self.axis_x)
        self.target_series.attachAxis(self.axis_y)
        self.fitted_series.attachAxis(self.axis_x)
        self.fitted_series.attachAxis(self.axis_y)

        # set title
        self.chart.setTitle(self.tr("Distribution Canvas"))
        # set labels
        self.axis_x.setTitleText(self.tr("Grain size")+" (μm)")
        self.axis_y.setTitleText(self.tr("Probability Density"))

        self.show_demo(self.axis_x, self.axis_y, x_log=True)
        self.component_series = [] # type: List[QtChart.QLineSeries]
        self.component_infinite_lines = [] # type: List[InfiniteLine]

    def on_component_number_changed(self, component_number: int):
        # use assert because the error must be handled in other place
        assert isinstance(component_number, int)
        assert component_number > 0
        # clear
        for series in self.component_series:
            self.chart.removeSeries(series)
        for line in self.component_infinite_lines:
            line.disconnect_from_chart()
        self.component_series.clear()
        self.component_infinite_lines.clear()
        # add
        for i in range(component_number):
            component_name = "C{0}".format(i+1)
            # series
            series = QtCharts.QLineSeries()
            series.setName(component_name)
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            # line
            line = InfiniteLine(is_horizontal=False, chart=self.chart,
                                pen=series.pen(), callback=self.on_infinite_line_moved)
            self.component_series.append(series)
            self.component_infinite_lines.append(line)
        # update the size of legend
        self.chart.legend().setMinimumSize(150.0, 30*(2+component_number))
        self.__current_component_number = component_number

    def show_target_distribution(self, sample: SampleData):
        # necessary to stop
        self.stop_demo()
        # update the title of canvas
        self.chart.setTitle(sample.name)
        self.target_series.replace(self.to_points(sample.classes, sample.distribution))
        self.fitted_series.clear()
        for series in self.component_series:
            series.clear()
        for line in self.component_infinite_lines:
            line.value = 1.0
        self.axis_x.setRange(sample.classes[0], sample.classes[-1])
        self.axis_y.setRange(0.0, np.max(sample.distribution)*1.2)

    def show_fitting_result(self, result: FittingResult, current_iteration=None):
        # necessary to stop
        self.stop_demo()
        # check the component number
        if self.__current_component_number != result.component_number:
            self.on_component_number_changed(result.component_number)
        # update the title
        if current_iteration is None:
            self.chart.setTitle(result.name)
        else:
            self.chart.setTitle(("{0} "+self.tr("Iteration")+" ({1})").format(
                result.name, current_iteration))
        # update target series
        self.target_series.replace(self.to_points(result.real_x, result.target_y))
        # update fitted series
        self.fitted_series.replace(self.to_points(result.real_x, result.fitted_y))
        # update component series
        for i, (component, series, line) in enumerate(zip(
                result.components,
                self.component_series,
                self.component_infinite_lines)):
            series.replace(self.to_points(result.real_x, component.component_y))
            # display the median (modal size) and fraction for users
            component_name = "C{0}".format(i+1)
            series.setName(component_name + " ({0:.1f} μm, {1:.1%})".format(component.median, component.fraction))
            if np.isnan(component.mean) or np.isinf(component.mean):
                # if mean is invalid, set the position of this line to initial position
                line.value = 1.0
            else:
                line.value = component.mean

    def on_fitting_epoch_suceeded(self, result: FittingResult):
        self.show_fitting_result(result)
        self.export_to_png("./images/distribution_canvas/png/{0} - {1} - {2}.png".format(
            result.name, result.distribution_type, result.component_number), pixel_ratio=2.0)
        self.export_to_svg("./images/distribution_canvas/svg/{0} - {1} - {2}.svg".format(
            result.name, result.distribution_type, result.component_number))

    def on_single_iteration_finished(self, current_iteration: int, result: FittingResult):
        self.show_fitting_result(result, current_iteration=current_iteration)

    def on_infinite_line_moved(self):
        # this method will be called by another object directly
        # use lock to keep safety
        self.__infinite_line_mutex.lock()
        expectedMeanValues = tuple(sorted([line.value for line in self.component_infinite_lines]))
        self.sigExpectedMeanValueChanged.emit(expectedMeanValues)
        self.__infinite_line_mutex.unlock()


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = DistributionCanvas(is_dark=False)
    canvas.chart.legend().hide()
    canvas.show()
    sys.exit(app.exec_())
