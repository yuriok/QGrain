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
        self.initUI()
        self.setAttribute(Qt.WA_StyledBackground, True)

    def initUI(self):
        self.mainLayout = QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.chart = QtCharts.QChart()
        self.chartView = QtCharts.QChartView()
        self.chartView.setChart(self.chart)
        self.mainLayout.addWidget(self.chartView)

    def setupChartStyle(self):
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
        self.chartView.setRenderHint(QPainter.Antialiasing)

    def setThemeMode(self, isDark: bool):
        if isDark:
            self.chart.setTheme(QtCharts.QChart.ChartThemeDark)
        else:
            self.chart.setTheme(QtCharts.QChart.ChartThemeLight)

    def toPoints(self, x: np.ndarray, y: np.ndarray):
        return [QPointF(x_value, y_value) for x_value, y_value in zip(x, y)]

    def showDemo(self, axisX: QtCharts.QAbstractAxis,
                 axisY: QtCharts.QAbstractAxis,
                 xLog=False, yLog=False):
        def love(x, a):
            return np.abs(x)**(2/3) + (0.9*np.sqrt(np.abs(3.3-x**2))) * np.sin(a*np.pi*x)
        series = QtCharts.QLineSeries()
        series.setName(self.tr("Demo"))
        a = 3.3
        x = np.linspace(-np.sqrt(3.3), np.sqrt(3.3), 1000)
        y = love(x, a)

        series.replace(self.toPoints(10**x if xLog else x, 10**y if yLog else y))
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
            series.replace(self.toPoints(10**x if xLog else x, 10**y if yLog else y))
        self.demoSeries = series
        self.demoTimer = QTimer()
        self.demoTimer.timeout.connect(update)
        self.demoTimer.start(1000/60)

    def stopDemo(self):
        if hasattr(self, "demoTimer"):
            self.demoTimer.stop()
            del self.demoTimer
        if hasattr(self, "demoSeries"):
            self.chart.removeSeries(self.demoSeries)
            del self.demoSeries

    def exportToPng(self, filename: str):
        image = QImage(self.chartView.width(),
                       self.chartView.height(),
                       QImage.Format_ARGB32)
        self.chartView.render(image)
        image.save(filename)

    def exportToSvg(self, filename: str):
        generator = QSvgGenerator()
        generator.setFileName(filename)
        self.chartView.render(generator)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    canvs = Canvas()
    canvs.chart.legend().hide()
    axisX = QtCharts.QValueAxis()
    axisY = QtCharts.QValueAxis()
    canvs.chart.addAxis(axisX, Qt.AlignBottom)
    canvs.chart.addAxis(axisY, Qt.AlignLeft)
    canvs.show()
    canvs.showDemo(axisX, axisY)
    # canvs.stopDemo()
    sys.exit(app.exec_())
