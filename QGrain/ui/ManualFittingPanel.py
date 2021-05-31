__all__ = ["ManualFittingPanel"]


# Obsolete Implement using QtCharts
"""
import copy
import typing
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import qtawesome as qta
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import (QCoreApplication, QEventLoop, QMutex, QPointF,
                            QRectF, QSizeF, QStandardPaths, Qt, QTimer, Signal)
from PySide2.QtGui import (QBrush, QColor, QFont, QFontMetrics, QPainter, QPen,
                           QPixmap)
from PySide2.QtSvg import QSvgGenerator
from PySide2.QtWidgets import (QAction, QDialog, QDoubleSpinBox, QFileDialog,
                               QGraphicsItem, QGraphicsScene,
                               QGraphicsSceneDragDropEvent,
                               QGraphicsSceneHoverEvent,
                               QGraphicsSceneMouseEvent, QGraphicsView,
                               QGridLayout, QGroupBox, QLabel, QMessageBox,
                               QPushButton, QSlider, QSpinBox, QSplitter,
                               QStyleOptionGraphicsItem, QWidget)
from QGrain.algorithms import DistributionType
from QGrain.algorithms.AsyncFittingWorker import AsyncFittingWorker
from QGrain.algorithms.distributions import (BaseDistribution,
                                             NormalDistribution,
                                             SkewNormalDistribution,
                                             WeibullDistribution)
from QGrain.statistic import logarithmic
from QGrain.models.FittingResult import FittingResult
from QGrain.models.FittingTask import FittingTask
from QGrain.models.GrainSizeSample import GrainSizeSample
from QGrain.models.MixedDistributionChartViewModel import MixedDistributionChartViewModel, get_demo_view_model
from QGrain.charts.MixedDistributionChart import MixedDistributionChart

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

class ChartExportDialog(QDialog):
    def __init__(self, canvas):
        super().__init__(parent=canvas)
        self.canvas = canvas
        self.file_dialog = QFileDialog(self)
        self.init_ui()
        self.setAttribute(Qt.WA_StyledBackground, True)

    def init_ui(self):
        self.main_layout = QGridLayout(self)
        self.preview_label = QLabel()
        self.preview_label.setFixedHeight(400)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.preview_label, 0, 0, 1, 4)
        self.width_label = QLabel(self.tr("Width"))
        self.width_input = QSpinBox()
        self.width_input.setRange(600, 1920)
        self.width_input.setSingleStep(10)
        self.width_input.setValue(600)
        self.main_layout.addWidget(self.width_label, 1, 0)
        self.main_layout.addWidget(self.width_input, 1, 1)

        self.height_label = QLabel(self.tr("Height"))
        self.height_input = QSpinBox()
        self.height_input.setRange(400, 1080)
        self.height_input.setSingleStep(10)
        self.height_input.setValue(400)
        self.main_layout.addWidget(self.height_label, 1, 2)
        self.main_layout.addWidget(self.height_input, 1, 3)
        self.pixel_ratio_label = QLabel(self.tr("Pixel Ratio"))
        self.pixel_ratio_input = QDoubleSpinBox()
        self.pixel_ratio_input.setRange(1.0, 10.0)
        self.pixel_ratio_input.setSingleStep(1.0)
        self.pixel_ratio_input.setValue(1.0)
        self.main_layout.addWidget(self.pixel_ratio_label, 2, 0)
        self.main_layout.addWidget(self.pixel_ratio_input, 2, 1)
        self.format_options = {
            self.tr("Scalable Vector Graphics (*.svg)"): "SVG",
            self.tr("Portable Network Graphics (*.png)"): "PNG",
            self.tr("Joint Photographic Experts Group (*.jpg)"): "JPG",
            self.tr("Windows Bitmap (*.bmp)"): "BMP",
            self.tr("Portable Pixmap (*.ppm)"): "PPM",
            self.tr("X11 Bitmap (*.xbm)"): "XBM",
            self.tr("X11 Pixmap (*.xpm)"): "XPM"}
        self.support_formats = [name for description, name in self.format_options.items()]
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.main_layout.addWidget(self.cancel_button, 3, 0, 1, 2)
        self.export_button = QPushButton(self.tr("Export"))
        self.main_layout.addWidget(self.export_button, 3, 2, 1, 2)

        self.width_input.valueChanged.connect(self.update)
        self.height_input.valueChanged.connect(self.update)
        self.pixel_ratio_input.valueChanged.connect(self.update)

        self.cancel_button.clicked.connect(self.close)
        self.export_button.clicked.connect(lambda: self.save_figure())


    @property
    def width(self) -> int:
        return self.width_input.value()

    @width.setter
    def width(self, value: int):
        assert isinstance(value, int)
        self.width_input.setValue(value)

    @property
    def height(self) -> int:
        return self.height_input.value()

    @height.setter
    def height(self, value: int):
        assert isinstance(value, int)
        self.height_input.setValue(value)

    @property
    def pixel_ratio(self) -> float:
        return self.pixel_ratio_input.value()

    @pixel_ratio.setter
    def pixel_ratio(self, value: Union[int, float, str]):
        assert isinstance(value, (int, float))
        self.pixel_ratio_input.setValue(value)

    def update(self):
        # save to settings
        pixmap = self.canvas.get_pixmap(width=self.width, height=self.height,
                                        pixel_ratio=self.pixel_ratio)
        preview_width = min(self.width, 800)
        preview_height = min(self.height, 600)
        preview_pixel_ratio = max(self.width*self.pixel_ratio/preview_width, self.height*self.pixel_ratio/preview_height)
        pixmap.setDevicePixelRatio(preview_pixel_ratio)
        self.preview_label.setPixmap(pixmap)

    def save_figure(self, filename: str = None):
        if filename is None:
            desktop_path = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)[0]
            filename, format_description = self.file_dialog.getSaveFileName(
                self, self.tr("Select Filename"),
                desktop_path, ";;".join([description for description, name in self.format_options.items()]))
            if filename is None or filename == "":
                return
            format_name = self.format_options[format_description]
        else:
            format_name = filename.split(".")[-1].upper()

        if format_name in ("BMP", "JPG", "PNG", "PPM", "XBM", "XPM"):
            self.canvas.export_pixmap(filename, width=self.width, height=self.height,
                                      pixel_ratio=self.pixel_ratio)
            self.last_format = format_name
        elif format_name == "SVG":
            self.canvas.export_svg(filename)
            self.last_format = format_name
        else:
            raise NotImplementedError(format_name)

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.main_layout = QGridLayout(self)
        # self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.chart = QtCharts.QChart()
        self.chart_view = QtCharts.QChartView()
        self.chart_view.setChart(self.chart)
        self.chart_view.setMinimumWidth(300)
        self.chart_view.setMinimumHeight(200)
        self.main_layout.addWidget(self.chart_view)

        self.export_dialog = ChartExportDialog(self)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        # self.edit_action = QAction(self.tr("Edit"))
        # self.addAction(self.edit_action)
        self.export_action = QAction(self.tr("Export"))
        self.export_action.triggered.connect(self.on_export_clicked)
        self.addAction(self.export_action)

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

    def to_points(self, x: np.ndarray, y: np.ndarray):
        return [QPointF(x_value, y_value) for x_value, y_value in zip(x, y)]

    def update_legend(self):
        # update the size of legend
        names = [series.name() for series in self.chart.series()]
        metrics = QFontMetrics(self.chart.legend().font(), self.chart_view)
        dpi = metrics.fontDpi()
        min_width = max([metrics.width(name) for name in names]) + dpi/2
        min_height = (len(names) + dpi/10) * metrics.height()
        self.chart.legend().setMinimumSize(min_width, min_height)


    def show_demo(self, axis_x: QtCharts.QAbstractAxis,
                 axis_y: QtCharts.QAbstractAxis,
                 x_log=False, y_log=False):
        def love(x, a):
            return np.abs(x)**(2/3) + (0.9*np.sqrt(np.abs(3.3-x**2))) * np.sin(a*np.pi*x)
        series = QtCharts.QLineSeries()
        series.setName(self.tr("Demo"))
        a = 3.3
        x = np.linspace(-np.sqrt(3.3), np.sqrt(3.3), 1000)
        y = love(x, a)

        series.replace(self.to_points(10**x if x_log else x, 10**y if y_log else y))
        self.chart.addSeries(series)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        scale = 1.2
        minX = -np.sqrt(3.3) * scale
        maxX = np.sqrt(3.3) * scale
        minY = -1.5737869944381024 * scale
        maxY = 2.367369351208529 * scale
        if x_log:
            axis_x.setRange(10**minX, 10**maxX)
        else:
            axis_x.setRange(minX, maxX)
        if y_log:
            axis_y.setRange(10**minY, 10**maxY)
        else:
            axis_y.setRange(minY, maxY)

        def update():
            nonlocal a
            a += 0.01
            if a > 33:
                a = 3.3
            y = love(x, a)
            series.replace(self.to_points(10**x if x_log else x, 10**y if y_log else y))
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

    def get_pixmap(self, width: Union[int] = 800, height: Union[int] = 600,
                   pixel_ratio: Union[int, float] = 1.0):
        geometry = self.chart_view.saveGeometry()
        assert isinstance(width, int)
        assert width > 0
        assert isinstance(height, int)
        assert height > 0
        assert isinstance(pixel_ratio, (int, float))
        # set geometry to make it in order
        self.chart_view.setGeometry(0, 0, width, height)
        image = QPixmap(int(width*pixel_ratio),
                       int(height*pixel_ratio))
        image.setDevicePixelRatio(pixel_ratio)
        self.chart_view.render(image)
        self.chart_view.restoreGeometry(geometry)
        return image

    def export_pixmap(self, filename: str,
                      width: Union[int] = 800,
                      height: Union[int] = 600,
                      pixel_ratio: Union[int, float] = 1.0):
        pixmap = self.get_pixmap(width=width, height=height, pixel_ratio=pixel_ratio)
        pixmap.save(filename)

    def export_svg(self, filename: str):
        geometry = self.chart_view.saveGeometry()
        # set geometry to make it in order
        self.chart_view.setGeometry(0, 0, 800, 600)
        generator = QSvgGenerator()
        generator.setFileName(filename)
        generator.setTitle("Generated by QGrain (version 0.3.3)")
        generator.setDescription("This svg image was generated by QGrain, you can take secondary process to make it can be published.")
        self.chart_view.render(generator)
        self.chart_view.restoreGeometry(geometry)

    def on_export_clicked(self):
        self.export_dialog.update()
        self.export_dialog.exec_()

class ManualFittingCanvas(Canvas):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_chart()

        self.setup_chart_style()
        self.chart.legend().detachFromChart()
        # refine high-dpi issue
        self.chart.legend().setPos(self.chart.plotArea().top() + 50, self.chart.plotArea().left() + 50.0)

    def init_chart(self):
        # init axes
        # self.axis_x = QtCharts.QLogValueAxis()
        # self.axis_x.setBase(10.0)
        self.axis_x = QtCharts.QValueAxis()
        self.axis_x.setMinorTickCount(-1)
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.axis_y = QtCharts.QValueAxis()
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)

        model = get_demo_view_model()
        # init two const series
        self.target_series = QtCharts.QScatterSeries()
        self.target_series.setName(self.tr("Target"))
        self.target_series.setMarkerSize(5.0)
        self.target_series.setPen(QPen(QColor("#000000")))
        self.target_series.setBrush(QColor("#000000"))
        self.chart.addSeries(self.target_series)
        self.mixed_series = QtCharts.QLineSeries()
        self.mixed_series.setName(self.tr("Mixed"))
        self.mixed_series.setPen(QPen(QColor("#000000"), 2.0))
        # self.mixed_series.setBrush(QColor(0, 0, 0))
        self.chart.addSeries(self.mixed_series)
        # attach series to axes
        self.target_series.attachAxis(self.axis_x)
        self.target_series.attachAxis(self.axis_y)
        self.mixed_series.attachAxis(self.axis_x)
        self.mixed_series.attachAxis(self.axis_y)

        # set title
        self.chart.setTitle(self.tr("Distribution Canvas"))
        # set labels
        self.axis_x.setTitleText(self.tr("Grain-size [φ]"))
        self.axis_y.setTitleText(self.tr("Frequency"))

        self.default_component_prefix = "C"
        self.component_series = [] # type: List[QtChart.QLineSeries]
        self.axis_x.setReverse(True)
        self.show_model(model)

    def on_n_components_changed(self, n_components: int):
        # clear
        for series in self.component_series:
            self.chart.removeSeries(series)
        self.component_series.clear()
        # add
        for i in range(n_components):
            component_name = f"{self.default_component_prefix}{i+1}"
            # series
            series = QtCharts.QLineSeries()
            series.setPen(QPen(QColor.fromRgbF(*plt.get_cmap()(i)), 2.0))
            series.nameChanged.connect(self.update_legend)
            series.setName(component_name)
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            self.component_series.append(series)

    @property
    def n_components(self) -> int:
        return len(self.component_series)

    @n_components.setter
    def n_components(self, value):
        assert isinstance(value, int)
        assert value > 0
        self.on_n_components_changed(value)

    def on_observe_iteration_changed(self, value: bool):
        self.observe_iteration_tag = value

    def show_model(self, model: MixedDistributionChartViewModel):
        if model.n_components != self.n_components:
            self.n_components = model.n_components
            # for series, color in zip(self.component_series):
            #     series.setPen(QPen(QColor(color), 2.0))

        # update the title of canvas
        self.chart.setTitle(model.title)
        self.target_series.replace(self.to_points(model.classes_φ, model.target))
        self.mixed_series.replace(self.to_points(model.classes_φ, model.mixed))
        for i, (series, distribution, fraction) in enumerate(zip(self.component_series, model.distributions, model.fractions)):
            series.replace(self.to_points(model.classes_φ, distribution*fraction))
            max_pos = np.unravel_index(np.argmax(distribution), distribution.shape)
            mode_φ = model.classes_φ[max_pos]
            mode_μm = 1000.0 * 2**(-mode_φ)
            series.setName(f"{model.component_prefix}{i+1} ({mode_μm:.1f} μm, {fraction:.1%})")
        self.axis_x.setRange(model.classes_φ[-1], model.classes_φ[0])
        self.axis_y.setRange(0.0, round(np.max(model.target)*1.2, 2))
"""

import copy

import numpy as np
import qtawesome as qta
from PySide2.QtCore import Qt, QTimer, Signal
from PySide2.QtWidgets import (QDialog, QDoubleSpinBox, QGridLayout, QGroupBox,
                               QLabel, QMessageBox, QPushButton, QSlider,
                               QSplitter)
from QGrain.algorithms import DistributionType
from QGrain.algorithms.AsyncFittingWorker import AsyncFittingWorker
from QGrain.algorithms.distributions import (BaseDistribution,
                                             NormalDistribution,
                                             SkewNormalDistribution,
                                             WeibullDistribution)
from QGrain.charts.MixedDistributionChart import MixedDistributionChart
from QGrain.ssu import SSUResult, SSUTask


class ManualFittingPanel(QDialog):
    manual_fitting_finished = Signal(SSUResult)
    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.Window)
        self.setWindowTitle(self.tr("SSU Manual Fitting Panel"))
        self.control_widgets = []
        self.input_widgets = []
        self.last_task = None
        self.last_result = None
        self.async_worker = AsyncFittingWorker()
        self.async_worker.background_worker.task_succeeded.connect(self.on_task_succeeded)
        self.initialize_ui()
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowFlags(Qt.Drawer)
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.update_chart)
        self.chart_timer.setSingleShot(True)

    def initialize_ui(self):
        self.main_layout = QGridLayout(self)

        self.chart_group = QGroupBox(self.tr("Chart"))
        self.chart_layout = QGridLayout(self.chart_group)
        self.chart = MixedDistributionChart(show_mode=True, toolbar=False)
        self.chart_layout.addWidget(self.chart)

        self.control_group = QGroupBox(self.tr("Control"))
        self.control_layout = QGridLayout(self.control_group)
        self.try_button = QPushButton(qta.icon("mdi.test-tube"), self.tr("Try"))
        self.try_button.clicked.connect(self.on_try_clicked)
        self.control_layout.addWidget(self.try_button, 1, 0, 1, 4)
        self.confirm_button = QPushButton(qta.icon("ei.ok-circle"), self.tr("Confirm"))
        self.confirm_button.clicked.connect(self.on_confirm_clicked)
        self.control_layout.addWidget(self.confirm_button, 2, 0, 1, 4)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.chart_group)
        self.splitter.addWidget(self.control_group)
        self.main_layout.addWidget(self.splitter)

    def change_n_components(self, n_components: int):
        for widget in self.control_widgets:
            self.control_layout.removeWidget(widget)
            widget.hide()
        self.control_widgets.clear()
        self.input_widgets.clear()

        widgets = []
        slider_range = (0, 1000)
        input_widgets = []
        mean_range = (-5, 15)
        std_range = (0.0, 10)
        weight_range = (0, 10)
        names = [self.tr("Mean"), self.tr("STD"), self.tr("Weight")]
        ranges = [mean_range, std_range, weight_range]
        slider_values = [500, 100, 100]
        input_values = [0.0, 1.0, 1.0]

        for i in range(n_components):
            group = QGroupBox(f"C{i+1}")
            group.setMinimumWidth(200)
            group_layout = QGridLayout(group)
            inputs = []
            for j, (name, range_, slider_value, input_value) in enumerate(zip(names, ranges, slider_values, input_values)):
                label = QLabel(name)
                slider = QSlider()
                slider.setRange(*slider_range)
                slider.setValue(slider_value)
                slider.setOrientation(Qt.Horizontal)
                input_ = QDoubleSpinBox()
                input_.setRange(*range_)
                input_.setDecimals(3)
                input_.setSingleStep(0.01)
                input_.setValue(input_value)
                slider.valueChanged.connect(self.on_value_changed)
                input_.valueChanged.connect(self.on_value_changed)
                slider.valueChanged.connect(lambda x, input_=input_, range_=range_: input_.setValue(x/1000*(range_[-1]-range_[0])+range_[0]))
                input_.valueChanged.connect(lambda x, slider=slider, range_=range_: slider.setValue((x-range_[0])/(range_[-1]-range_[0])*1000))

                group_layout.addWidget(label, j, 0)
                group_layout.addWidget(slider, j, 1)
                group_layout.addWidget(input_, j, 2)
                inputs.append(input_)

            self.control_layout.addWidget(group, i+5, 0, 1, 4)
            widgets.append(group)
            input_widgets.append(inputs)

        self.control_widgets = widgets
        self.input_widgets = input_widgets

    @property
    def n_components(self) -> int:
        return len(self.input_widgets)

    @property
    def expected(self):
        reference = []
        weights = []
        for i, (mean, std, weight) in enumerate(self.input_widgets):
            reference.append(dict(mean=mean.value(), std=std.value(), skewness=0.0))
            weights.append(weight.value())
        weights = np.array(weights)
        fractions = weights / np.sum(weights)
        return reference, fractions

    def show_message(self, title: str, message: str):
        self.msg_box.setWindowTitle(title)
        self.msg_box.setText(message)
        self.msg_box.exec_()

    def show_info(self, message: str):
        self.show_message(self.tr("Info"), message)

    def show_warning(self, message: str):
        self.show_message(self.tr("Warning"), message)

    def show_error(self, message: str):
        self.show_message(self.tr("Error"), message)

    def on_confirm_clicked(self):
        if self.last_result is not None:
            for component, (mean, std, weight) in zip(self.last_result.components, self.input_widgets):
                mean.setValue(component.logarithmic_moments["mean"])
                std.setValue(component.logarithmic_moments["std"])
                weight.setValue(component.fraction*10)
            self.manual_fitting_finished.emit(self.last_result)

            self.last_result = None
            self.last_task = None
            self.try_button.setEnabled(False)
            self.confirm_button.setEnabled(False)
            self.hide()

    def on_task_failed(self, info: str, task: SSUTask):
        self.show_error(info)

    def on_task_succeeded(self, result: SSUResult):
        self.chart.show_model(result.view_model)
        self.last_result = result
        self.confirm_button.setEnabled(True)

    def on_try_clicked(self):
        if self.last_task is None:
            return
        new_task = copy.copy(self.last_task)
        reference, fractions = self.expected
        initial_guess = BaseDistribution.get_initial_guess(self.last_task.distribution_type, reference, fractions=fractions)
        new_task.initial_guess = initial_guess
        self.async_worker.execute_task(new_task)

    def on_value_changed(self):
        self.chart_timer.stop()
        self.chart_timer.start(10)

    def update_chart(self):
        if self.last_task is None:
            return
        reference, fractions = self.expected
        for comp_ref in reference:
            if comp_ref["std"] == 0.0:
                return
        # print(reference)
        initial_guess = BaseDistribution.get_initial_guess(self.last_task.distribution_type, reference, fractions=fractions)
        result = SSUResult(self.last_task, initial_guess)
        self.chart.show_model(result.view_model, quick=True)

    def setup_task(self, task: SSUTask):
        self.last_task = task
        self.try_button.setEnabled(True)
        if self.n_components != task.n_components:
            self.change_n_components(task.n_components)
        reference, fractions = self.expected
        initial_guess = BaseDistribution.get_initial_guess(task.distribution_type, reference, fractions=fractions)
        result = SSUResult(task, initial_guess)
        self.chart.show_model(result.view_model, quick=False)


if __name__ == "__main__":
    import sys

    from QGrain.entry import setup_app
    from QGrain.artificial import get_random_sample

    app, splash = setup_app()
    main = ManualFittingPanel()
    main.show()
    splash.finish(main)
    sample = get_random_sample().sample_to_fit
    task = SSUTask(sample, DistributionType.Normal, 3)
    main.setup_task(task)
    sys.exit(app.exec_())
