import logging
import random

import numpy as np
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import Qt
from PySide2.QtGui import QColor, QPen

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())
from models.SampleData import SampleData
from models.SampleDataset import SampleDataset
from ui.Canvas import Canvas


class AllDistributionCanvas(Canvas):
    logger = logging.getLogger("root.ui.AllDistributionCanvas")
    gui_logger = logging.getLogger("GUI")

    def __init__(self, parent=None, isDark=True):
        super().__init__(parent)
        self.setThemeMode(isDark)
        self.initChart()
        self.setupChartStyle()
        self.chart.legend().hide()

    def initChart(self):
        # init axes
        self.axisX = QtCharts.QLogValueAxis()
        self.axisX.setBase(10.0)
        self.axisX.setMinorTickCount(-1)
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.axisY = QtCharts.QValueAxis()
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        # set title
        self.chart.setTitle(self.tr("All Distribution Canvas"))
        # set labels
        self.axisX.setTitleText(self.tr("Grain size")+" (μm)")
        self.axisY.setTitleText(self.tr("Probability Density"))

        self.showDemo(self.axisX, self.axisY, xLog=True)

    def on_data_loaded(self, dataset: SampleDataset):
        self.stopDemo()
        self.chart.removeAllSeries()
        maxY = -1.0
        for sample in dataset.samples:
            series = QtCharts.QLineSeries()
            series.setName(sample.name)
            series.replace(self.toPoints(sample.classes, sample.distribution))
            self.chart.addSeries(series)
            pen = QPen(QColor(series.pen().color()),
                       1.0)
            series.setPen(pen)
            series.attachAxis(self.axisX)
            series.attachAxis(self.axisY)
            currentMaxY = np.max(sample.distribution)
            if currentMaxY > maxY:
                maxY = currentMaxY
        self.axisX.setRange(dataset.classes[0], dataset.classes[-1])
        self.axisY.setRange(0.0, maxY*1.2)

        self.exportToPng("./temp/all_distribution_canvas.png")
        self.exportToSvg("./temp/all_distribution_canvas.svg")


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = AllDistributionCanvas(isDark=False)
    canvas.chart.legend().hide()
    canvas.show()
    sys.exit(app.exec_())
