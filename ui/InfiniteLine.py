from typing import Optional, Union, Tuple, Callable

from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QPointF, QRectF, QSizeF, Qt
from PySide2.QtGui import QColor, QPainter, QPen
from PySide2.QtWidgets import (QGraphicsItem, QGraphicsSceneHoverEvent,
                               QGraphicsSceneMouseEvent,
                               QStyleOptionGraphicsItem, QWidget)


class InfiniteLine(QGraphicsItem):
    def __init__(self, isHorizontal: bool, chart: QtCharts.QChart,
                 pen: Optional[QPen] = None, hoverPen: Optional[QPen] = None,
                 callback: Callable = None):
        super().__init__()
        self.__isHorizontal = isHorizontal
        self.__chart = chart
        self.__callback = callback

        self.__pen = QPen(QColor(0x000000), 3.0) if pen is None else pen
        self.__hoverPen = QPen(QColor(0xff0000), 5.0) if hoverPen is None else hoverPen
        self.__currentPen = self.__pen

        self.__value = None
        self.value = sum(self.valueRange) / 2
        assert self.__value is not None

        self.setAcceptHoverEvents(True)
        self.connectToChart(chart)

    def boundingRect(self) -> QRectF:
        w = self.__currentPen.widthF()
        if self.isHorizontal:
            return QRectF(QPointF(0.0, -w/2), QSizeF(self.length, w*2))
        else:
            return QRectF(QPointF(-w/2, 0.0), QSizeF(w*2, self.length))

    def paint(self, painter: QPainter, style: QStyleOptionGraphicsItem, widget: QWidget):
        painter.save()
        painter.setPen(self.__currentPen)
        # painter.setRenderHint(QPainter.Antialiasing)
        w = self.__currentPen.widthF()
        if self.isHorizontal:
            painter.drawLine(w/2, 0.0, self.length-w/2, 0.0)
        else:
            painter.drawLine(0.0, w/2, 0.0, self.length-w/2)
        painter.restore()

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        self.__currentPen = self.hoverPen
        self.update()
        return True

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        self.__currentPen = self.pen
        self.update()
        return True

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self.isHorizontal:
            new = self.__chart.mapToValue(event.scenePos()).y()
        else:
            new = self.__chart.mapToValue(event.scenePos()).x()
        self.value = new
        # if the value has been changed by the user
        if self.__callback is not None:
            self.__callback()

    def mousePressEvent(self, event): pass

    def mouseReleaseEvent(self, event): pass

    @property
    def isHorizontal(self) -> bool:
        return self.__isHorizontal

    @isHorizontal.setter
    def isHorizontal(self, value: bool):
        if isinstance(value, bool):
            self.__isHorizontal = value
        else:
            raise TypeError(value)

    @property
    def length(self) -> float:
        if self.isHorizontal:
            return self.__chart.plotArea().width()
        else:
            return self.__chart.plotArea().height()

    @property
    def valueRange(self) -> Tuple[float, float]:
        top_left = self.__chart.mapToValue(self.__chart.plotArea().topLeft())
        bottom_right = self.__chart.mapToValue(
            self.__chart.plotArea().bottomRight())
        if self.isHorizontal:
            left, right = top_left.y(), bottom_right.y()
        else:
            left, right = top_left.x(), bottom_right.x()
        if left < right:
            return left, right
        else:
            return right, left

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, chart_value: Union[int, float]):
        left, right = self.valueRange
        if chart_value < left or chart_value > right:
            return
        if self.isHorizontal:
            pos = self.__chart.mapToPosition(QPointF(0.0, chart_value))
            self.setY(pos.y())
        else:
            pos = self.__chart.mapToPosition(QPointF(chart_value, 0.0))
            self.setX(pos.x())
        self.__value = chart_value

    @property
    def pen(self) -> QPen:
        return self.__pen

    @pen.setter
    def pen(self, value: QPen):
        if isinstance(value, QPen):
            self.__pen = value
        else:
            raise TypeError(value)

    @property
    def hoverPen(self) -> QPen:
        return self.__hoverPen

    @hoverPen.setter
    def hoverPen(self, value: QPen):
        if isinstance(value, QPen):
            self.__hoverPen = value
        else:
            raise TypeError(value)

    def connectToChart(self, chart: QtCharts.QChart):
        self.__chart = chart
        chart.scene().addItem(self)
        chart.plotAreaChanged.connect(self.onPlotAreaChanged)
        self.onPlotAreaChanged(chart.plotArea())

    def disconnectFromChart(self):
        self.__chart.scene().removeItem(self)
        self.__chart.plotAreaChanged.disconnect(self.onPlotAreaChanged)

    def onPlotAreaChanged(self, plotArea: QRectF):
        if self.isHorizontal:
            self.setX(plotArea.left())
        else:
            self.setY(plotArea.top())
        self.value = self.value
