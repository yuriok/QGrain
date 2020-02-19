import math
import sys
from typing import Optional, Union

import numpy as np
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QLocale, QPointF, QRectF, QSizeF, Qt, QTimer
from PySide2.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPainter, QImage
from PySide2.QtWidgets import (QApplication, QColorDialog, QGraphicsItem,
                               QGraphicsScene, QGraphicsSceneDragDropEvent,
                               QGraphicsSceneHoverEvent,
                               QGraphicsSceneMouseEvent, QGraphicsView,
                               QGridLayout, QMainWindow, QPushButton,
                               QStyleOptionGraphicsItem, QWidget, QSizePolicy)
from PySide2.QtSvg import QSvgGenerator

class Canvas(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self.init_ui()
        self.setAttribute(Qt.WA_StyledBackground, True)

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.chart = QtCharts.QChart()
        self.chart_view = QtCharts.QChartView()
        self.chart_view.setChart(self.chart)
        self.main_layout.addWidget(self.chart_view)

    def setup_chart_style(self):
        self.chart.setTitleFont(QFont("Times New Roman", 12))
        # self.chart.setTitleBrush(QBrush(QColor(0xff0000))) # set color
        self.chart.axisX().setTitleFont(QFont("Times New Roman", 10))
        # self.chart.axisX().setTitleBrush(QBrush(QColor(0xff0000)))
        self.chart.axisY().setTitleFont(QFont("Times New Roman", 10))
        # self.chart.axisX().setTitleBrush(QBrush(QColor(0xff0000)))
        self.chart.legend().setAlignment(Qt.AlignTop)
        self.chart.legend().setMarkerShape(QtCharts.QLegend.MarkerShapeFromSeries)
        self.chart.legend().setFont(QFont("Times New Roman", 10))
        # self.chart.setAnimationOptions(QtCharts.QChart.AllAnimations)
        self.chart.setBackgroundVisible(False)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

    def set_theme_mode(self, isDark: bool):
        if isDark:
            self.chart.setTheme(QtCharts.QChart.ChartThemeDark)
        else:
            self.chart.setTheme(QtCharts.QChart.ChartThemeLight)

    def to_points(self, x: np.ndarray, y: np.ndarray):
        return [QPointF(x_value, y_value) for x_value, y_value in zip(x, y)]

    def show_demo(self, axisX: QtCharts.QAbstractAxis,
                 axisY: QtCharts.QAbstractAxis,
                 xLog=False, yLog=False):
        def love(x, a):
            return np.abs(x)**(2/3) + (0.9*np.sqrt(np.abs(3.3-x**2))) * np.sin(a*np.pi*x)
        series = QtCharts.QLineSeries()
        series.setName(self.tr("Demo"))
        a = 3.3
        x = np.linspace(-np.sqrt(3.3), np.sqrt(3.3), 1000)
        y = love(x, a)

        series.replace(self.to_points(10**x if xLog else x, 10**y if yLog else y))
        self.chart.addSeries(series)
        series.attachAxis(axisX)
        series.attachAxis(axisY)
        scale = 1.2
        minX = -np.sqrt(3.3) * scale
        maxX = np.sqrt(3.3) * scale
        minY = -1.5737869944381024 * scale
        maxY = 2.367369351208529 * scale
        if xLog:
            axisX.setRange(10**minX, 10**maxX)
        else:
            axisX.setRange(minX, maxX)
        if yLog:
            axisY.setRange(10**minY, 10**maxY)
        else:
            axisY.setRange(minY, maxY)

        def update():
            nonlocal a
            a += 0.01
            if a > 33:
                a = 3.3
            y = love(x, a)
            series.replace(self.to_points(10**x if xLog else x, 10**y if yLog else y))
        self.demo_series = series
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(update)
        self.demo_timer.start(1000/60)

    def stop_demo(self):
        if hasattr(self, "demo_timer"):
            self.demo_timer.stop()
            del self.demo_timer
        if hasattr(self, "demo_series"):
            self.chart.removeSeries(self.demo_series)
            del self.demo_series

    def export_to_png(self, filename: str, pixel_ration: Union[int, float] = 1.0):
        geometry = self.chart_view.saveGeometry()
        # set geometry to make it in order
        self.chart_view.setGeometry(0, 0, 800, 600)
        if not isinstance(pixel_ration, (int, float)):
            raise TypeError(pixel_ration)
        if pixel_ration <= 0:
            raise ValueError(pixel_ration)
        image = QImage(int(self.chart_view.width()*pixel_ration),
                       int(self.chart_view.height()*pixel_ration),
                       QImage.Format_ARGB32)
        image.setDevicePixelRatio(pixel_ration)
        self.chart_view.render(image)
        image.save(filename)
        self.chart_view.restoreGeometry(geometry)

    def export_to_svg(self, filename: str):
        geometry = self.chart_view.saveGeometry()
        # set geometry to make it in order
        self.chart_view.setGeometry(0, 0, 800, 600)
        generator = QSvgGenerator()
        generator.setFileName(filename)
        self.chart_view.render(generator)
        self.chart_view.restoreGeometry(geometry)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    canvs = Canvas()
    canvs.chart.legend().hide()
    axisX = QtCharts.QValueAxis()
    axisY = QtCharts.QValueAxis()
    canvs.chart.addAxis(axisX, Qt.AlignBottom)
    canvs.chart.addAxis(axisY, Qt.AlignLeft)
    canvs.show()
    canvs.show_demo(axisX, axisY)
    # canvs.stopDemo()
    sys.exit(app.exec_())
