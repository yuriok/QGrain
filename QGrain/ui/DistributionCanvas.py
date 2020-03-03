__all__ = ["DistributionCanvas"]

import logging
from typing import List

import cv2
import numpy as np
from PIL import Image
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import (QCoreApplication, QEventLoop, QMutex,
                            QStandardPaths, Qt, QTimer, Signal)
from PySide2.QtGui import QColor, QPen
from PySide2.QtWidgets import QFileDialog

from QGrain.models.FittingResult import FittingResult
from QGrain.models.SampleData import SampleData
from QGrain.ui.Canvas import Canvas
from QGrain.ui.InfiniteLine import InfiniteLine


class DistributionCanvas(Canvas):
    sigExpectedMeanValueChanged = Signal(tuple)
    logger = logging.getLogger("root.ui.DistributionCanvas")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, is_dark=True):
        super().__init__("DistributionCanvas", parent=parent)
        self.set_theme_mode(is_dark)
        self.init_chart()
        self.setup_chart_style()
        self.chart.legend().detachFromChart()
        # refine high-dpi issue
        self.chart.legend().setPos(self.chart.plotArea().top() + 50, self.chart.plotArea().left() + 50.0)
        self.__infinite_line_mutex = QMutex()
        self.__iteration_mutex = QMutex()
        self.__iteration_timer = QTimer()
        # used to check if it's necessary to update the component series and lines
        self.__current_component_number = None
        self.video_format_options = {self.tr("MP4 (*.mp4)"): "mp4v",
                                     self.tr("AVI Motion-JPEG (*.avi)"): "MJPG",
                                     self.tr("AVI MPEG-4.2 (*.avi)"): "MP42",
                                     self.tr("AVI MPEG-4.3 (*.avi)"): "DIV3",
                                     self.tr("AVI MPEG-4 (*.avi)"): "DIVX"}
        self.file_dialog = QFileDialog(self)

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
        self.observe_iteration_tag = False

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
            series.nameChanged.connect(self.update_legend)
            series.setName(component_name)
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            # line
            line = InfiniteLine(is_horizontal=False, chart=self.chart,
                                pen=series.pen(), callback=self.on_infinite_line_moved)
            self.component_series.append(series)
            self.component_infinite_lines.append(line)
        self.__current_component_number = component_number

    def on_observe_iteration_changed(self, value: bool):
        self.observe_iteration_tag = value

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
        self.axis_y.setRange(0.0, round(np.max(sample.distribution)*1.2, 2))

    def __update_components(self, result: FittingResult):
        # update fitted series
        self.fitted_series.replace(self.to_points(result.real_x, result.fitted_y))
        # update component series
        for component_index, (component, series, line) in enumerate(zip(
                result.components,
                self.component_series,
                self.component_infinite_lines)):
            series.replace(self.to_points(result.real_x, component.component_y))
            # display the median (modal size) and fraction for users
            component_name = "C{0}".format(component_index+1)
            series.setName(component_name + " ({0:.1f} μm, {1:.1%})".format(component.median, component.fraction))
            if np.isnan(component.mean) or np.isinf(component.mean):
                # if mean is invalid, set the position of this line to initial position
                line.value = 1.0
            else:
                line.value = component.mean
        self.axis_x.setRange(result.real_x[0], result.real_x[-1])
        self.axis_y.setRange(0.0, round(np.max(result.target_y)*1.2, 2))

    def show_fitting_result(self, result: FittingResult):
        # necessary to stop
        self.stop_demo()
        success = self.__iteration_mutex.tryLock()
        if not success:
            return
        # update target series
        self.target_series.replace(self.to_points(result.real_x, result.target_y))
        # check the component number
        if self.__current_component_number != result.component_number:
            self.on_component_number_changed(result.component_number)

        generator = result.history
        current_iteration = 0

        def closure():
            try:
                nonlocal current_iteration
                result_in_process = next(generator)
                self.chart.setTitle(("{0} "+self.tr("Iteration")+" ({1})").format(
                    result.name, current_iteration))
                self.__update_components(result_in_process)
                current_iteration += 1
            except StopIteration:
                self.__iteration_timer.stop()
                self.__iteration_timer.timeout.disconnect(closure)
                self.chart.setTitle(result.name)
                self.__update_components(result)
                self.__iteration_mutex.unlock()

        if self.observe_iteration_tag:
            self.__iteration_timer.timeout.connect(closure)
            self.__iteration_timer.start(1000//30)
        else:
            self.chart.setTitle(result.name)
            self.__update_components(result)
            self.__iteration_mutex.unlock()

    def generate_video(self, result: FittingResult):
        # necessary to stop
        self.stop_demo()
        # update target series
        self.target_series.replace(self.to_points(result.real_x, result.target_y))
        # check the component number
        if self.__current_component_number != result.component_number:
            self.on_component_number_changed(result.component_number)
        width = self.export_dialog.width
        height = self.export_dialog.height
        pixel_ratio = self.export_dialog.pixel_ratio
        video_size = (int(width*pixel_ratio), int(height*pixel_ratio))
        fps = 30.0
        desktop_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
        filename, format_name = self.file_dialog.getSaveFileName(
            self, self.tr("Select Filename"),
            desktop_path, ";;".join([format_name for format_name, fourcc in self.video_format_options.items()]))
        video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*self.video_format_options[format_name]), fps, video_size)
        for current_iteration, result_in_process in enumerate(result.history):
            self.chart.setTitle(("{0} "+self.tr("Iteration")+" ({1})").format(
                    result.name, current_iteration))
            self.__update_components(result_in_process)
            QCoreApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            pixmap = self.get_pixmap(width=width, height=height, pixel_ratio=pixel_ratio)
            mat = cv2.cvtColor(np.asarray(Image.fromqpixmap(pixmap)), cv2.COLOR_RGB2BGR)
            video.write(mat)

        video.release()

    def on_infinite_line_moved(self):
        # this method will be called by another object directly
        # use lock to keep safety
        self.__infinite_line_mutex.lock()
        expected_mean_values = tuple(sorted([line.value for line in self.component_infinite_lines]))
        self.sigExpectedMeanValueChanged.emit(expected_mean_values)
        self.__infinite_line_mutex.unlock()


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = DistributionCanvas(is_dark=False)
    canvas.on_component_number_changed(3)
    canvas.chart.legend().hide()
    canvas.show()
    sys.exit(app.exec_())
