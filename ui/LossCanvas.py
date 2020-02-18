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

    def __init__(self, parent=None, isDark=True):
        super().__init__(parent)
        self.setThemeMode(isDark)
        self.initChart()
        self.setupChartStyle()
        self.chart.legend().hide()

    def initChart(self):
        # init axes
        self.axisX = QtCharts.QValueAxis()
        self.axisX.setTickInterval(20)
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.axisY = QtCharts.QLogValueAxis()
        self.axisY.setBase(10.0)
        # self.axisY.setMinorTickCount(-1)
        self.axisY.setLabelFormat("%.1e")
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        # init the series
        self.lossSeries = QtCharts.QLineSeries()
        self.lossSeries.setName(self.tr("Loss"))
        self.chart.addSeries(self.lossSeries)
        # attach series to axes
        self.lossSeries.attachAxis(self.axisX)
        self.lossSeries.attachAxis(self.axisY)
        # set title
        self.chart.setTitle(self.tr("Loss Canvas"))
        # set labels
        self.axisX.setTitleText(self.tr("Iteration"))
        self.axisY.setTitleText(self.tr("Loss"))

        self.showDemo(self.axisX, self.axisY, yLog=True)
        # data
        self.result_info = None
        self.max_loss = -sys.maxsize
        self.min_loss = sys.maxsize

    def on_fitting_started(self):
        self.lossSeries.clear()
        self.result_info = None
        self.max_loss = -sys.maxsize
        self.min_loss = sys.maxsize

    def on_fitting_finished(self):
        if self.result_info is None:
            return
        name, distribution_type, component_number = self.result_info
        self.chart.setTitle(name)
        self.exportToPng("./temp/loss_canvas/png/{0} - {1} - {2}.png".format(
            name, distribution_type, component_number))
        self.exportToSvg("./temp/loss_canvas/svg/{0} - {1} - {2}.svg".format(
            name, distribution_type, component_number))

    def on_single_iteration_finished(self, current_iteration: int, result: FittingResult):
        if current_iteration == 0:
            # necessary to stop
            self.stopDemo()
            self.result_info = (result.name, result.distribution_type, result.component_number)
        self.chart.setTitle(("{0} "+self.tr("Iteration")+" ({1})").format(result.name, current_iteration))
        loss = result.mean_squared_error
        self.lossSeries.append(current_iteration, loss)
        self.axisX.setRange(0.0, current_iteration)
        if loss > self.max_loss:
            self.max_loss = loss
        if loss < self.min_loss:
            self.min_loss = loss
        self.axisY.setRange(self.min_loss, self.max_loss)


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = LossCanvas(isDark=False)
    canvas.chart.legend().hide()
    canvas.show()
    sys.exit(app.exec_())
