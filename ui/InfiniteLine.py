from typing import Optional, Union, Tuple, Callable

from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QPointF, QRectF, QSizeF, Qt
from PySide2.QtGui import QColor, QPainter, QPen
from PySide2.QtWidgets import (QGraphicsItem, QGraphicsSceneHoverEvent,
                               QGraphicsSceneMouseEvent,
                               QStyleOptionGraphicsItem, QWidget)


class InfiniteLine(QGraphicsItem):
    def __init__(self, is_horizontal: bool, chart: QtCharts.QChart,
                 pen: Optional[QPen] = None, hover_pen: Optional[QPen] = None,
                 callback: Callable = None):
        super().__init__()
        self.__is_horizontal = is_horizontal
        self.__chart = chart
        self.__callback = callback

        self.__pen = QPen(QColor(0x000000), 3.0) if pen is None else pen
        self.__hover_pen = QPen(QColor(0xff0000), 5.0) if hover_pen is None else hover_pen
        self.__current_pen = self.__pen

        self.__value = None
        self.value = sum(self.value_range) / 2
        assert self.__value is not None

        self.setAcceptHoverEvents(True)
        self.connect_to_chart(chart)

    def boundingRect(self) -> QRectF:
        w = self.__current_pen.widthF()
        if self.is_horizontal:
            return QRectF(QPointF(0.0, -w/2), QSizeF(self.length, w*2))
        else:
            return QRectF(QPointF(-w/2, 0.0), QSizeF(w*2, self.length))

    def paint(self, painter: QPainter, style: QStyleOptionGraphicsItem, widget: QWidget):
        painter.save()
        painter.setPen(self.__current_pen)
        # painter.setRenderHint(QPainter.Antialiasing)
        w = self.__current_pen.widthF()
        if self.is_horizontal:
            painter.drawLine(w/2, 0.0, self.length-w/2, 0.0)
        else:
            painter.drawLine(0.0, w/2, 0.0, self.length-w/2)
        painter.restore()

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        self.__current_pen = self.hover_pen
        self.update()
        return True

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        self.__current_pen = self.pen
        self.update()
        return True

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self.is_horizontal:
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
    def is_horizontal(self) -> bool:
        return self.__is_horizontal

    @is_horizontal.setter
    def is_horizontal(self, value: bool):
        if isinstance(value, bool):
            self.__is_horizontal = value
        else:
            raise TypeError(value)

    @property
    def length(self) -> float:
        if self.is_horizontal:
            return self.__chart.plotArea().width()
        else:
            return self.__chart.plotArea().height()

    @property
    def value_range(self) -> Tuple[float, float]:
        top_left = self.__chart.mapToValue(self.__chart.plotArea().topLeft())
        bottom_right = self.__chart.mapToValue(
            self.__chart.plotArea().bottomRight())
        if self.is_horizontal:
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
        left, right = self.value_range
        if chart_value < left or chart_value > right:
            return
        if self.is_horizontal:
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
    def hover_pen(self) -> QPen:
        return self.__hover_pen

    @hover_pen.setter
    def hover_pen(self, value: QPen):
        if isinstance(value, QPen):
            self.__hover_pen = value
        else:
            raise TypeError(value)

    def connect_to_chart(self, chart: QtCharts.QChart):
        self.__chart = chart
        chart.scene().addItem(self)
        chart.plotAreaChanged.connect(self.on_plot_area_changed)
        self.on_plot_area_changed(chart.plotArea())

    def disconnect_from_chart(self):
        self.__chart.scene().removeItem(self)
        self.__chart.plotAreaChanged.disconnect(self.on_plot_area_changed)

    def on_plot_area_changed(self, plot_area: QRectF):
        if self.is_horizontal:
            self.setX(plot_area.left())
        else:
            self.setY(plot_area.top())
        self.value = self.value
