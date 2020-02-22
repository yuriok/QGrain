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

        self.show_demo(self.axis_x, self.axis_y, xLog=True)
        self.component_series = [] # type: List[QtChart.QLineSeries]
        self.component_infinite_lines = [] # type: List[InfiniteLine]

    def on_component_number_changed(self, component_number: int):
        self.logger.info("Received the component changed signal, start to clear and add data items.")
        # Check the validity of `component_number`
        if type(component_number) != int:
            raise TypeError(component_number)
        if component_number <= 0:
            raise ValueError(component_number)
        # clear
        for series in self.component_series:
            self.chart.removeSeries(series)
        for line in self.component_infinite_lines:
            line.disconnect_from_chart()
        self.component_series.clear()
        self.component_infinite_lines.clear()
        self.logger.debug("Items cleared.")
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
        self.logger.debug("Items added.")

    def on_target_data_changed(self, sample: SampleData):
        # necessary to stop
        self.stop_demo()
        # update the title of canvas
        if sample.name is None or sample.name == "":
            sample.name = "UNKNOWN"
        self.chart.setTitle(sample.name)
        self.target_series.replace(self.to_points(sample.classes, sample.distribution))
        self.fitted_series.clear()
        for series in self.component_series:
            series.clear()
        for line in self.component_infinite_lines:
            line.value = 1.0
        self.axis_x.setRange(sample.classes[0], sample.classes[-1])
        self.axis_y.setRange(0.0, np.max(sample.distribution)*1.2)
        self.logger.debug("Target data has been changed to [%s].", sample.name)

    def update_canvas_by_data(self, result: FittingResult, current_iteration=None):
        # necessary to stop
        self.stop_demo()
        # update the title of canvas
        if current_iteration is not None:
            self.chart.setTitle(("{0} "+self.tr("Iteration")+" ({1})").format(
                result.name, current_iteration))
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
        self.update_canvas_by_data(result)
        self.export_to_png("./images/distribution_canvas/png/{0} - {1} - {2}.png".format(
            result.name, result.distribution_type, result.component_number), pixel_ratio=2.0)
        self.export_to_svg("./images/distribution_canvas/svg/{0} - {1} - {2}.svg".format(
            result.name, result.distribution_type, result.component_number))

    def on_single_iteration_finished(self, current_iteration: int, result: FittingResult):
        self.update_canvas_by_data(result, current_iteration=current_iteration)

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
