import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout, QWidget
from QGrain.algorithms.moments import get_SSC_proportion
from QGrain.models.GrainSizeSample import GrainSizeSample

class SSCDiagramChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Sand-silt-clay Diagram"))
        self.figure = plt.figure()
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()
        self.draw_base()

    def trans_pos(self, silt, sand):
        silt, sand = np.array(silt), np.array(sand)
        y = 0.5 * np.sqrt(3) * sand
        # calculate the cross of two lines
        # first line: y = sand
        # second line: the line cross the two points, (clay, 0) and (0.5, 0.5*np.sqrt(3))
        # y = ax + b
        key = np.not_equal(silt, 0.5)
        x = np.full_like(y, np.nan)
        a = np.full_like(y, np.nan)
        b = np.full_like(y, np.nan)
        x[~key] = 0.5
        a[key] = 0.5*np.sqrt(3) / (0.5-silt[key])
        b[key] = -a[key] * silt[key]
        x[key] = (y[key] - b[key]) / a[key]
        assert not np.any(np.isnan(x))
        return x, y

    def draw_base(self):
        self.axes.axis("off")
        self.axes.set_aspect(1)
        self.axes.set_title(self.tr("Sand-silt-clay Diagram"))
        structural_lines = [([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]), # the 3 sides of this equilateral triangle
                            ([0.0, 1.0], [0.1, 0.1]), # sand = 10%
                            ([0.0, 1.0], [0.5, 0.5]), # sand = 50%
                            ([0.0, 1.0], [0.9, 0.9]), # sand = 90%
                            ([1/3, 1/3], [0.0, 0.9]), # clay: silt = 1:2
                            ([2/3, 2/3], [0.0, 0.9])] # clay: silt = 2:1
        span = 0.01
        labels = [((-span, -span), self.tr("Clay"), dict(ha="right", va="top", fontweight="bold")),
                  ((0.5, 1+span), self.tr("Sand"), dict(ha="center", va="bottom", fontweight="bold")),
                  ((1+span, -span), self.tr("Silt"), dict(ha="left", va="top", fontweight="bold")),
                  ((-5*span, 0.55), self.tr("Sand %"), dict(ha="right", va="center", fontweight="bold")),
                  ((0.5, -3*span), self.tr("Silt:Clay Ratio"), dict(ha="center", va="top", fontweight="bold")),

                  ((0.16, 0.05), self.tr("Clay"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.5, 0.05), self.tr("Mud"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.84, 0.05), self.tr("Silt"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.16, 0.3), self.tr("Sandy Clay"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.5, 0.3), self.tr("Sandy Mud"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.84, 0.3), self.tr("Sandy Silt"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.16, 0.55), self.tr("Clayey Sand"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.5, 0.55), self.tr("Muddy Sand"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.84, 0.55), self.tr("Silty Sand"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.5, 0.91), self.tr("Sand"), dict(ha="center", va="bottom", fontsize="x-small")),

                  ((-1*span, 0.1), self.tr("10%"), dict(ha="right", va="center", fontsize="x-small")),
                  ((-5*span, 0.5), self.tr("50%"), dict(ha="right", va="center", fontsize="x-small")),
                  ((-9*span, 0.9), self.tr("90%"), dict(ha="right", va="center", fontsize="x-small")),

                  ((1/3, -span), self.tr("1:2"), dict(ha="center", va="top", fontsize="x-small")),
                  ((2/3, -span), self.tr("2:1"), dict(ha="center", va="top", fontsize="x-small"))]

        for sand, clay in structural_lines:
            x, y = self.trans_pos(sand, clay)
            self.axes.plot(x, y, linewidth=0.8, c="black")

        for (sand, clay), text, kwargs in labels:
            x, y = self.trans_pos(sand, clay)
            self.axes.text(x, y, text, **kwargs)
        self.figure.tight_layout()

    def show_samples(self, samples: typing.Iterable[GrainSizeSample], append=False):
        if not append:
            self.axes.clear()
            self.draw_base()
        silt = []
        sand = []
        for i, sample in enumerate(samples):
            sand_i, silt_i, clay_i = get_SSC_proportion(sample.classes_φ, sample.distribution)
            silt.append(silt_i)
            sand.append(sand_i)
        x, y = self.trans_pos(silt, sand)
        self.axes.plot(x, y, c="#ffffff00", marker=".", ms=8, mfc="black", mew=0.0)[0]
        self.canvas.draw()


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication, QStyleFactory
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setStyleSheet("""* {font-family:Tahoma,Verdana,Arial,Georgia,"Microsoft YaHei","Times New Roman";}""")
    chart = SSCDiagramChart()
    chart.draw_base()
    chart.show()
    sys.exit(app.exec_())
