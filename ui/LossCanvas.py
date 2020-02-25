import logging
import sys

from PySide2.QtCharts import QtCharts
from PySide2.QtCore import Qt


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())
from models.FittingResult import FittingResult
from ui.Canvas import Canvas


class LossCanvas(Canvas):
    logger = logging.getLogger("root.ui.FittingCanvas")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, is_dark=True):
        super().__init__(parent)
        self.set_theme_mode(is_dark)
        self.init_chart()
        self.setup_chart_style()
        self.chart.legend().hide()

    def init_chart(self):
        # init axes
        self.axis_x = QtCharts.QValueAxis()
        self.axis_x.setTickInterval(20)
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.axis_y = QtCharts.QLogValueAxis()
        self.axis_y.setBase(10.0)
        self.axis_y.setLabelFormat("%.1e")
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        # init the series
        self.loss_series = QtCharts.QLineSeries()
        self.loss_series.setName(self.tr("Loss"))
        self.chart.addSeries(self.loss_series)
        # attach series to axes
        self.loss_series.attachAxis(self.axis_x)
        self.loss_series.attachAxis(self.axis_y)
        # set title
        self.chart.setTitle(self.tr("Loss Canvas"))
        # set labels
        self.axis_x.setTitleText(self.tr("Iteration"))
        self.axis_y.setTitleText(self.tr("Loss"))

        self.show_demo(self.axis_x, self.axis_y, y_log=True)
        # data
        self.result_info = None
        self.max_loss = -sys.maxsize
        self.min_loss = sys.maxsize

    def on_fitting_started(self):
        self.loss_series.clear()
        self.result_info = None
        self.max_loss = -sys.maxsize
        self.min_loss = sys.maxsize

    def on_fitting_finished(self):
        if self.result_info is None:
            return
        name, distribution_type, component_number = self.result_info
        self.chart.setTitle(name)
        self.export_to_png("./images/loss_canvas/png/{0} - {1} - {2}.png".format(
            name, distribution_type, component_number))
        self.export_to_svg("./images/loss_canvas/svg/{0} - {1} - {2}.svg".format(
            name, distribution_type, component_number))

    def on_single_iteration_finished(self, current_iteration: int, result: FittingResult):
        if current_iteration == 0:
            # necessary to stop
            self.stop_demo()
            self.result_info = (result.name, result.distribution_type, result.component_number)
        self.chart.setTitle(("{0} "+self.tr("Iteration")+" ({1})").format(result.name, current_iteration))
        loss = result.mean_squared_error
        self.loss_series.append(current_iteration, loss)
        self.axis_x.setRange(0.0, current_iteration)
        if loss > self.max_loss:
            self.max_loss = loss
        if loss < self.min_loss:
            self.min_loss = loss
        self.axis_y.setRange(self.min_loss, self.max_loss)


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = LossCanvas(is_dark=False)
    canvas.chart.legend().hide()
    canvas.show()
    sys.exit(app.exec_())
