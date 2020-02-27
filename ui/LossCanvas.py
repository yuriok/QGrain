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
        self.observe_iteration_tag = False

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

    def on_observe_iteration_changed(self, value: bool):
        self.observe_iteration_tag = value

    def show_fitting_result(self, result: FittingResult):
        if not self.observe_iteration_tag:
            return
        self.stop_demo()
        self.loss_series.clear()
        max_loss = -sys.maxsize
        min_loss = sys.maxsize
        for i, result_in_process in enumerate(result.history):
            self.loss_series.append(i, result_in_process.mean_squared_error)
            if result_in_process.mean_squared_error > max_loss:
                max_loss = result_in_process.mean_squared_error
            if result_in_process.mean_squared_error < min_loss:
                min_loss = result_in_process.mean_squared_error
            # self.chart.setTitle(("{0} "+self.tr("Iteration")+" ({1})").format(result.name, i))
            # self.axis_x.setRange(0.0, current_iteration)
            # self.axis_y.setRange(min_Loss, max_loss)

        self.axis_x.setRange(0, result.iteration_number-1)
        self.axis_y.setRange(min_loss, max_loss)
        self.chart.setTitle(result.name)
        self.export_pixmap("./images/loss_canvas/png/{0} - {1} - {2}.png".format(
            result.name, result.distribution_type, result.component_number))
        self.export_svg("./images/loss_canvas/svg/{0} - {1} - {2}.svg".format(
            result.name, result.distribution_type, result.component_number))


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = LossCanvas(is_dark=False)
    canvas.chart.legend().hide()
    canvas.show()
    sys.exit(app.exec_())
