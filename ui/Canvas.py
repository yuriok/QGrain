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
        self.initUi()
        self.setAttribute(Qt.WA_StyledBackground, True)

    def initUi(self):
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

    def setThemeMode(self, isDark: bool):
        if isDark:
            self.chart.setTheme(QtCharts.QChart.ChartThemeDark)
        else:
            self.chart.setTheme(QtCharts.QChart.ChartThemeLight)

    def toPoints(self, x: np.ndarray, y: np.ndarray):
        return [QPointF(x_value, y_value) for x_value, y_value in zip(x, y)]

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
