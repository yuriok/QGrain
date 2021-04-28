import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout
from QGrain.algorithms.moments import get_GSM_proportion
from QGrain.models.GrainSizeSample import GrainSizeSample

class GSMDiagramChart(QDialog):
    def __init__(self, parent=None, toolbar=False):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.setWindowTitle(self.tr("Gravel-sand-mud Diagram"))
        self.figure = plt.figure(figsize=(6, 6))
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()

        self.draw_base()

    def trans_pos(self, sand, gravel):
        sand, gravel = np.array(sand), np.array(gravel)
        y = 0.5 * np.sqrt(3) * gravel
        # calculate the cross of two lines
        # first line: y = sand
        # second line: the line cross the two points, (clay, 0) and (0.5, 0.5*np.sqrt(3))
        key = np.not_equal(sand, 0.5)
        x = np.full_like(y, np.nan)
        a = np.full_like(y, np.nan)
        b = np.full_like(y, np.nan)
        x[~key] = 0.5
        a[key] = 0.5*np.sqrt(3) / (0.5-sand[key])
        b[key] = -a[key] * sand[key]
        x[key] = (y[key] - b[key]) / a[key]
        assert not np.any(np.isnan(x))
        return x, y

    def draw_base(self):
        self.axes.axis("off")
        self.axes.set_aspect(1)
        self.axes.set_title(self.tr("Gravel-sand-mud Diagram"))
        structural_lines = [([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]), # the 3 sides of this equilateral triangle
                            ([0.0, 1.0], [0.01, 0.01]), # gravel = Trace
                            ([0.0, 1.0], [0.05, 0.05]), # gravel = 5%
                            ([0.0, 1.0], [0.3, 0.3]), # gravel = 30%
                            ([0.0, 1.0], [0.8, 0.8]), # gravel = 80%

                            ([1/10, 1/10], [0.0, 0.05]), # sand: mud = 1:9
                            ([1/2, 1/2], [0.0, 0.8]), # sand: mud = 1:1
                            ([9/10, 9/10], [0.0, 0.8]), # sand: mud = 9:1

                            # addtional lines
                            ([0.05, -0.03], [0.02, 0.12]),
                            ([0.3, 0.3], [0.02, 0.06]),
                            ([0.7, 0.7], [0.02, 0.06]),
                            ([0.95, 1.03], [0.02, 0.12]),
                            ([0.95, 1.03], [0.02, 0.12]),
                            ([0.95, 1.03], [0.2, 0.275]),
                            ([0.95, 1.03], [0.5, 0.55])]


        span = 0.01
        labels = [((-span, -span), self.tr("Mud"), dict(ha="right", va="top", fontweight="bold")),
                  ((0.5, 1+span), self.tr("Gravel"), dict(ha="center", va="bottom", fontweight="bold")),
                  ((1+span, -span), self.tr("Sand"), dict(ha="left", va="top", fontweight="bold")),
                  ((-5*span, 0.55), self.tr("Gravel %"), dict(ha="right", va="center", fontweight="bold")),
                  ((0.5, -5*span), self.tr("Sand:Mud Ratio"), dict(ha="center", va="top", fontweight="bold")),

                  ((0.05, -span), self.tr("Mud"), dict(ha="center", va="top", fontsize="x-small")),
                  ((0.3, -span), self.tr("Sandy Mud"), dict(ha="center", va="top", fontsize="x-small")),
                  ((0.7, -span), self.tr("Muddy Sand"), dict(ha="center", va="top", fontsize="x-small")),
                  ((0.95, -span), self.tr("Sand"), dict(ha="center", va="top", fontsize="x-small")),

                  ((-0.03, 0.12), self.tr("Slightly\nGravelly\nMud"), dict(ha="right", va="bottom", fontsize="x-small")),
                  ((0.3, 0.08), self.tr("Slightly Gravelly Sandy Mud"), dict(ha="center", va="bottom", fontsize="x-small")),
                  ((0.7, 0.08), self.tr("Slightly Gravelly Muddy Sand"), dict(ha="center", va="bottom", fontsize="x-small")),
                  ((1.03, 0.12), self.tr("Slightly\nGravelly\nSand"), dict(ha="left", va="bottom", fontsize="x-small")),

                  ((0.2, 0.2), self.tr("Gravelly Mud"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.7, 0.2), self.tr("Gravelly Muddy Sand"), dict(ha="center", va="center", fontsize="x-small")),
                  ((1.03, 0.28), self.tr("Gravelly\nSand"), dict(ha="left", va="bottom", fontsize="x-small")),
                  ((0.2, 0.55), self.tr("Muddy\nGravel"), dict(ha="center", va="center", fontsize="x-small")),
                  ((0.7, 0.55), self.tr("Muddy\nSandy\nGravel"), dict(ha="center", va="center", fontsize="x-small")),
                  ((1.03, 0.55), self.tr("Sandy\nGravel"), dict(ha="left", va="bottom", fontsize="x-small")),
                  ((0.5, 0.85), self.tr("Gravel"), dict(ha="center", va="bottom", fontsize="x-small")),

                  ((-1*span, 0.01), self.tr("Trace"), dict(ha="right", va="center", fontsize="x-small")),
                  ((-1*span, 0.05), self.tr("5%"), dict(ha="right", va="center", fontsize="x-small")),
                  ((-3*span, 0.3), self.tr("30%"), dict(ha="right", va="center", fontsize="x-small")),
                  ((-8*span, 0.8), self.tr("80%"), dict(ha="right", va="center", fontsize="x-small")),

                  ((1/10, -span), self.tr("1:9"), dict(ha="center", va="top", fontsize="x-small")),
                  ((1/2, -span), self.tr("1:1"), dict(ha="center", va="top", fontsize="x-small")),
                  ((9/10, -span), self.tr("9:1"), dict(ha="center", va="top", fontsize="x-small"))]

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
        sand = []
        gravel = []
        for i, sample in enumerate(samples):
            gravel_i, sand_i, mud_i = get_GSM_proportion(sample.classes_φ, sample.distribution)
            sand.append(sand_i)
            gravel.append(gravel_i)
        x, y = self.trans_pos(sand, gravel)
        self.axes.plot(x, y, c="#ffffff00", marker=".", ms=8, mfc="black", mew=0.0)[0]
        self.canvas.draw()


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication, QStyleFactory
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setStyleSheet("""* {font-family:Tahoma,Verdana,Arial,Georgia,"Microsoft YaHei","Times New Roman";}""")
    chart = GSMDiagramChart()
    chart.draw_base()
    chart.show()
    sys.exit(app.exec_())
