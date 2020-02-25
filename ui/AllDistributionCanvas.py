import logging

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

    def __init__(self, parent=None, is_dark=True):
        super().__init__(parent)
        self.set_theme_mode(is_dark)
        self.init_chart()
        self.setup_chart_style()
        self.chart.legend().hide()

    def init_chart(self):
        # init axes
        self.axis_x = QtCharts.QLogValueAxis()
        self.axis_x.setBase(10.0)
        self.axis_x.setMinorTickCount(-1)
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.axis_y = QtCharts.QValueAxis()
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        # set title
        self.chart.setTitle(self.tr("All Distribution Canvas"))
        # set labels
        self.axis_x.setTitleText(self.tr("Grain size")+" (μm)")
        self.axis_y.setTitleText(self.tr("Probability Density"))

        self.show_demo(self.axis_x, self.axis_y, x_log=True)

    def on_data_loaded(self, dataset: SampleDataset):
        self.stop_demo()
        self.chart.removeAllSeries()
        max_y = -1.0
        for sample in dataset.samples:
            series = QtCharts.QLineSeries()
            series.setName(sample.name)
            series.replace(self.to_points(sample.classes, sample.distribution))
            self.chart.addSeries(series)
            pen = QPen(QColor(series.pen().color()),
                       1.0)
            series.setPen(pen)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            current_max_y = np.max(sample.distribution)
            if current_max_y > max_y:
                max_y = current_max_y
        self.axis_x.setRange(dataset.classes[0], dataset.classes[-1])
        self.axis_y.setRange(0.0, max_y*1.2)

        self.export_to_png("./images/all_distribution_canvas.png", pixel_ratio=5.0)
        self.export_to_svg("./images/all_distribution_canvas.svg")


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = AllDistributionCanvas(is_dark=False)
    canvas.chart.legend().hide()
    canvas.show()
    sys.exit(app.exec_())
