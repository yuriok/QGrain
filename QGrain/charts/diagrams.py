import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QGridLayout
from QGrain.algorithms.moments import get_GSM_proportion, get_SSC_proportion
from QGrain.models.GrainSizeSample import GrainSizeSample


class DiagramChart(QDialog):
    def __init__(self, parent=None, toolbar=False, figsize=(8, 6)):
        flags = Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint
        super().__init__(parent=parent, f=flags)
        self.figure = plt.figure(figsize=figsize)
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.main_layout.addWidget(self.canvas, 1, 0, 1, 2)
        if not toolbar:
            self.toolbar.hide()
        self.draw_base()

    def trans_pos(self, a, b):
        pass

    @property
    def title(self):
        pass

    @property
    def lines(self):
        pass

    @property
    def labels(self):
        pass


    def plot_legend(self):
        pass

    def draw_base(self):
        self.axes.axis("off")
        self.axes.set_aspect(1)
        self.axes.set_title(self.title)

        for sand, clay, kwargs in self.lines:
            x, y = self.trans_pos(sand, clay)
            self.axes.plot(x, y, **kwargs)

        for (sand, clay), text, kwargs in self.labels:
            x, y = self.trans_pos(sand, clay)
            self.axes.text(x, y, text, **kwargs)

        self.plot_legend()
        self.figure.tight_layout()

    def convert_samples(self, samples: typing.List[GrainSizeSample]):
        pass

    def show_samples(self, samples: typing.List[GrainSizeSample],
                     append=False, c="#ffffff00",
                     marker=".", ms=8, mfc="black", mew=0.0,
                     **kwargs):
        if not append:
            self.axes.clear()
            self.draw_base()
        a, b = self.convert_samples(samples)
        x, y = self.trans_pos(a, b)
        self.axes.plot(x, y, c=c, marker=marker, ms=ms, mfc=mfc, mew=mew, **kwargs)
        self.canvas.draw()

class Folk54GSMDiagramChart(DiagramChart):
    def __init__(self, parent=None, toolbar=False):
        super().__init__(parent=parent, toolbar=toolbar)
        self.setWindowTitle(self.tr("Gravel-sand-mud diagram (Folk, 1954)"))

    @property
    def title(self):
        return self.tr("Gravel-sand-mud diagram (Folk, 1954)")

    @property
    def lines(self):
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c="black", linewidth=0.8)), # the 3 sides of this equilateral triangle
            ([0.0, 1.0], [0.01, 0.01], dict(c="black", linewidth=0.8)), # gravel = Trace
            ([0.0, 1.0], [0.05, 0.05], dict(c="black", linewidth=0.8)), # gravel = 5%
            ([0.0, 1.0], [0.3, 0.3], dict(c="black", linewidth=0.8)), # gravel = 30%
            ([0.0, 1.0], [0.8, 0.8], dict(c="black", linewidth=0.8)), # gravel = 80%

            ([1/10, 1/10], [0.0, 0.05], dict(c="black", linewidth=0.8)), # sand: mud = 1:9
            ([1/2, 1/2], [0.0, 0.8], dict(c="black", linewidth=0.8)), # sand: mud = 1:1
            ([9/10, 9/10], [0.0, 0.8], dict(c="black", linewidth=0.8))] # sand: mud = 9:1
        ADDTIONAL_LINES = [
            ([0.05, -0.03], [0.02, 0.12], dict(c="black", linewidth=0.8)),
            ([0.3, 0.3], [0.02, 0.06], dict(c="black", linewidth=0.8)),
            ([0.7, 0.7], [0.02, 0.06], dict(c="black", linewidth=0.8)),
            ([0.95, 1.03], [0.02, 0.12], dict(c="black", linewidth=0.8)),
            ([0.95, 1.03], [0.02, 0.12], dict(c="black", linewidth=0.8)),
            ([0.95, 1.03], [0.2, 0.275], dict(c="black", linewidth=0.8)),
            ([0.95, 1.03], [0.5, 0.55], dict(c="black", linewidth=0.8))]

        return STRUCTURAL_LINES + ADDTIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [((-s, -s), self.tr("Mud"), dict(ha="right", va="top", fontweight="bold")),
                  ((0.5, 1+s), self.tr("Gravel"), dict(ha="center", va="bottom", fontweight="bold")),
                  ((1+s, -s), self.tr("Sand"), dict(ha="left", va="top", fontweight="bold")),
                  ((-5*s, 0.55), self.tr("Gravel %"), dict(ha="right", va="center", fontweight="bold")),
                  ((0.5, -5*s), self.tr("Sand:Mud Ratio"), dict(ha="center", va="top", fontweight="bold")),

                  ((0.05, -s), self.tr("Mud"), dict(ha="center", va="top", fontsize="x-small")),
                  ((0.3, -s), self.tr("Sandy Mud"), dict(ha="center", va="top", fontsize="x-small")),
                  ((0.7, -s), self.tr("Muddy Sand"), dict(ha="center", va="top", fontsize="x-small")),
                  ((0.95, -s), self.tr("Sand"), dict(ha="center", va="top", fontsize="x-small")),

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

                  ((-1*s, 0.01), self.tr("Trace"), dict(ha="right", va="center", fontsize="x-small")),
                  ((-1*s, 0.05), self.tr("5%"), dict(ha="right", va="center", fontsize="x-small")),
                  ((-3*s, 0.3), self.tr("30%"), dict(ha="right", va="center", fontsize="x-small")),
                  ((-8*s, 0.8), self.tr("80%"), dict(ha="right", va="center", fontsize="x-small")),

                  ((1/10, -s), self.tr("1:9"), dict(ha="center", va="top", fontsize="x-small")),
                  ((1/2, -s), self.tr("1:1"), dict(ha="center", va="top", fontsize="x-small")),
                  ((9/10, -s), self.tr("9:1"), dict(ha="center", va="top", fontsize="x-small"))]

        return LABELS

    def trans_pos(self, sand, gravel):
        sand, gravel = np.array(sand), np.array(gravel)
        y = 0.5 * np.sqrt(3) * gravel
        # calculate the cross of two lines
        # first line: y = 0.5 * np.sqrt(3) * gravel
        # second line: the line cross the two points, (sand, 0) and (0.5, 0.5*np.sqrt(3))
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

    def convert_samples(self, samples: typing.List[GrainSizeSample]):
        sand = []
        gravel = []
        for i, sample in enumerate(samples):
            gravel_i, sand_i, mud_i = get_GSM_proportion(sample.classes_φ, sample.distribution)
            sand.append(sand_i)
            gravel.append(gravel_i)
        return sand, gravel

class Folk54SSCDiagramChart(DiagramChart):
    def __init__(self, parent=None, toolbar=False):
        super().__init__(parent=parent, toolbar=toolbar)
        self.setWindowTitle(self.tr("Sand-silt-clay diagram (Folk, 1954)"))

    @property
    def title(self):
        return self.tr("Sand-silt-clay diagram (Folk, 1954)")

    @property
    def lines(self):
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c="black", linewidth=0.8)), # the 3 sides of this equilateral triangle
            ([0.0, 1.0], [0.1, 0.1], dict(c="black", linewidth=0.8)), # sand = 10%
            ([0.0, 1.0], [0.5, 0.5], dict(c="black", linewidth=0.8)), # sand = 50%
            ([0.0, 1.0], [0.9, 0.9], dict(c="black", linewidth=0.8)), # sand = 90%
            ([1/3, 1/3], [0.0, 0.9], dict(c="black", linewidth=0.8)), # clay: silt = 1:2
            ([2/3, 2/3], [0.0, 0.9], dict(c="black", linewidth=0.8))] # clay: silt = 2:1
        ADDTIONAL_LINES = []

        return STRUCTURAL_LINES + ADDTIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), self.tr("Clay"), dict(ha="right", va="top", fontweight="bold")),
            ((0.5, 1+s), self.tr("Sand"), dict(ha="center", va="bottom", fontweight="bold")),
            ((1+s, -s), self.tr("Silt"), dict(ha="left", va="top", fontweight="bold")),
            ((-5*s, 0.55), self.tr("Sand %"), dict(ha="right", va="center", fontweight="bold")),
            ((0.5, -3*s), self.tr("Silt:Clay Ratio"), dict(ha="center", va="top", fontweight="bold")),

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

            ((-1*s, 0.1), self.tr("10%"), dict(ha="right", va="center", fontsize="x-small")),
            ((-5*s, 0.5), self.tr("50%"), dict(ha="right", va="center", fontsize="x-small")),
            ((-9*s, 0.9), self.tr("90%"), dict(ha="right", va="center", fontsize="x-small")),

            ((1/3, -s), self.tr("1:2"), dict(ha="center", va="top", fontsize="x-small")),
            ((2/3, -s), self.tr("2:1"), dict(ha="center", va="top", fontsize="x-small"))]

        return LABELS

    def trans_pos(self, silt, sand):
        silt, sand = np.array(silt), np.array(sand)
        y = 0.5 * np.sqrt(3) * sand
        # calculate the cross of two lines
        # first line: 0.5 * np.sqrt(3) * sand
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

    def convert_samples(self, samples: typing.List[GrainSizeSample]):
        silt = []
        sand = []
        for i, sample in enumerate(samples):
            sand_i, silt_i, clay_i = get_SSC_proportion(sample.classes_φ, sample.distribution)
            silt.append(silt_i)
            sand.append(sand_i)
        return silt, sand

class BP12GSMDiagramChart(DiagramChart):
    def __init__(self, parent=None, toolbar=False):
        super().__init__(parent=parent, toolbar=toolbar)
        self.setWindowTitle(self.tr("Gravel-sand-mud diagram (Blott & Pye, 2012)"))

    @property
    def title(self):
        return self.tr("Gravel-sand-mud diagram (Blott & Pye, 2012)")

    @property
    def lines(self):
        span = 0.01
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c="black", linewidth=0.8)), # the 3 sides of this equilateral triangle
            # Gravel
            ([0.0, 0.99], [0.01, 0.01], dict(c="black", linewidth=0.8)), # gravel = Trace
            ([0.0, 0.95], [0.05, 0.05], dict(c="black", linewidth=0.8)), # gravel = 5%
            ([0.0, 0.8], [0.2, 0.2], dict(c="black", linewidth=0.8)), # gravel = 20%
            ([0.0, 1/3, 0.5], [0.5, 1/3, 0.5], dict(c="black", linewidth=0.8)), # gravel = 50%, 33%
            # Sand
            ([0.01, 0.01], [0.0, 0.99], dict(c="black", linewidth=0.8)), # sand = Trace
            ([0.05, 0.05], [0.0, 0.95], dict(c="black", linewidth=0.8)), # sand = 5%
            ([0.2, 0.2], [0.0, 0.8], dict(c="black", linewidth=0.8)), # sand = 20%
            ([0.5, 1/3, 0.5], [0.0, 1/3, 0.5], dict(c="black", linewidth=0.8)), # sand = 50%, 33%
            # Mud
            ([0.99, 0.0], [0.0, 0.99], dict(c="black", linewidth=0.8)), # mud = Trace
            ([0.95, 0.0], [0.0, 0.95], dict(c="black", linewidth=0.8)), # mud = 5%
            ([0.8, 0.0], [0.0, 0.8], dict(c="black", linewidth=0.8)), # mud = 20%
            ([0.5, 1/3, 0.0], [0.0, 1/3, 0.5], dict(c="black", linewidth=0.8))] # mud = 50%, 33%
        ADDITIONAL_LINES = [
            ([0.0, -span], [0.0, -span], dict(c="black", linewidth=0.4)), # M
            ([0.0, -2*span], [1.0, 1+4*span], dict(c="black", linewidth=0.4)), # G
            ([1.0, 1+2*span], [0.0, -span], dict(c="black", linewidth=0.4)), # S
            # ticks of Gravel
            ([0.0, -span], [1.0, 1.0+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.9, 0.9+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.8, 0.8+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.7, 0.7+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.6, 0.6+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.5, 0.5+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.4, 0.4+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.3, 0.3+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.2, 0.2+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.1, 0.1+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.0, span], dict(c="black", linewidth=0.4)),
            # ticks of Sand
            ([0.0, 0.0+0.1*span], [1.0, 1.0+span], dict(c="black", linewidth=0.4)),
            ([0.1, 0.1+0.1*span], [0.9, 0.9+span], dict(c="black", linewidth=0.4)),
            ([0.2, 0.2+0.1*span], [0.8, 0.8+span], dict(c="black", linewidth=0.4)),
            ([0.3, 0.3+0.1*span], [0.7, 0.7+span], dict(c="black", linewidth=0.4)),
            ([0.4, 0.4+0.1*span], [0.6, 0.6+span], dict(c="black", linewidth=0.4)),
            ([0.5, 0.5+0.1*span], [0.5, 0.5+span], dict(c="black", linewidth=0.4)),
            ([0.6, 0.6+0.1*span], [0.4, 0.4+span], dict(c="black", linewidth=0.4)),
            ([0.7, 0.7+0.1*span], [0.3, 0.3+span], dict(c="black", linewidth=0.4)),
            ([0.8, 0.8+0.1*span], [0.2, 0.2+span], dict(c="black", linewidth=0.4)),
            ([0.9, 0.9+0.1*span], [0.1, 0.1+span], dict(c="black", linewidth=0.4)),
            ([1.0, 1.0+0.1*span], [0.0, span], dict(c="black", linewidth=0.4)),
            # ticks of Mud
            ([0.0, span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.1, 0.1+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.2, 0.2+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.3, 0.3+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.4, 0.4+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.5, 0.5+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.6, 0.6+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.7, 0.7+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.8, 0.8+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.9, 0.9+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([1.0, 1.0-span], [0.0, -span], dict(c="black", linewidth=0.4)),

            # guide lines
            ([0.03, 0.03+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.13, 0.13+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.33, 0.33+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.63, 0.63+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.83, 0.83+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.97, 0.97], [0.0, -2*span], dict(c="black", linewidth=0.4)),

            ([0.0, -2*span], [0.03, 0.03+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.13, 0.13+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.33, 0.33+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.63, 0.63+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.83, 0.83+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.97, 0.97], dict(c="black", linewidth=0.4)),

            ([0.97, 0.97+0.2*span], [0.03, 0.03+2*span], dict(c="black", linewidth=0.4)),
            ([0.87, 0.87+0.2*span], [0.13, 0.13+2*span], dict(c="black", linewidth=0.4)),
            ([0.67, 0.67+0.2*span], [0.33, 0.33+2*span], dict(c="black", linewidth=0.4)),
            ([0.37, 0.37+0.2*span], [0.63, 0.63+2*span], dict(c="black", linewidth=0.4)),
            ([0.17, 0.17+0.2*span], [0.83, 0.83+2*span], dict(c="black", linewidth=0.4)),
            ([0.03, 0.03+0.2*span], [0.97, 0.97+2*span], dict(c="black", linewidth=0.4)),

            ([0.03, 0.03+2*span], [0.95, 0.95+2*span], dict(c="black", linewidth=0.4)),
            ([0.03, -2*span], [0.03, 0.03+4*span], dict(c="black", linewidth=0.4)),
            ([0.94, 0.94+0.4*span], [0.03, 0.03+4*span], dict(c="black", linewidth=0.4)),

            ]
        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), self.tr("M"), dict(ha="right", va="top", fontweight="bold")),
            ((-2*s, 1+4*s), self.tr("G"), dict(ha="center", va="bottom", fontweight="bold")),
            ((1+2*s, -s), self.tr("S"), dict(ha="left", va="top", fontweight="bold")),
            ((-5*s, 0.55), self.tr("Gravel %"), dict(ha="right", va="center", fontweight="bold")),
            ((0.5, 0.55), self.tr("Sand %"), dict(ha="left", va="center", fontweight="bold")),
            ((0.5+5*s, -5*np.sqrt(3)*s), self.tr("Mud %"), dict(ha="center", va="top", fontweight="bold")),
            # tick labels of Gravel
            ((-s, 1.0+s), self.tr("100%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.9+s), self.tr("90%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.8+s), self.tr("80%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.7+s), self.tr("70%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.6+s), self.tr("60%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.5+s), self.tr("50%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.4+s), self.tr("40%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.3+s), self.tr("30%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.2+s), self.tr("20%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.1+s), self.tr("10%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.0+s), self.tr("0%"), dict(ha="right", va="bottom", fontsize="x-small")),
            # tick labels of Sand
            ((0.0, 1.0+s), self.tr("0%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.1, 0.9+s), self.tr("10%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.2, 0.8+s), self.tr("20%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.3, 0.7+s), self.tr("30%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.4, 0.6+s), self.tr("40%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.5, 0.5+s), self.tr("50%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.6, 0.4+s), self.tr("60%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.7, 0.3+s), self.tr("70%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.8, 0.2+s), self.tr("80%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.9, 0.1+s), self.tr("90%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((1.0, 0.0+s), self.tr("100%"), dict(ha="left", va="bottom", fontsize="x-small")),
            # tick labels of Mud
            ((2*s, -s), self.tr("100%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.1+2*s, -s), self.tr("90%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.2+2*s, -s), self.tr("80%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.3+2*s, -s), self.tr("70%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.4+2*s, -s), self.tr("60%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.5+2*s, -s), self.tr("50%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.6+2*s, -s), self.tr("40%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.7+2*s, -s), self.tr("30%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.8+2*s, -s), self.tr("20%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.9+2*s, -s), self.tr("10%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((1.0-s, -s), self.tr("0%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),

            ((0.125, 0.025), self.tr("(vg)(s)M"), dict(ha="center", va="center", fontsize="small")),
            ((0.35, 0.025), self.tr("(vg)sM"), dict(ha="center", va="center", fontsize="small")),
            ((0.625, 0.025), self.tr("(vg)mS"), dict(ha="center", va="center", fontsize="small")),
            ((0.850, 0.025), self.tr("(vg)(m)S"), dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.125), self.tr("(g)(s)M"), dict(ha="center", va="center", fontsize="small")),
            ((0.3, 0.125), self.tr("(g)sM"), dict(ha="center", va="center", fontsize="small")),
            ((0.575, 0.125), self.tr("(g)mS"), dict(ha="center", va="center", fontsize="small")),
            ((0.750, 0.125), self.tr("(g)(m)S"), dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.35), self.tr("(s)gM"), dict(ha="center", va="center", fontsize="small")),
            ((0.3, 0.3), self.tr("gsM"), dict(ha="center", va="center", fontsize="small")),
            ((0.4, 0.3), self.tr("gmS"), dict(ha="center", va="center", fontsize="small")),
            ((0.525, 0.35), self.tr("(m)gS"), dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.55), self.tr("(s)mG"), dict(ha="center", va="center", fontsize="small")),
            ((1/3-(0.45-1/3)/2, 0.45), self.tr("smG"), dict(ha="center", va="center", fontsize="small")),
            ((0.325, 0.55), self.tr("(m)sG"), dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.75), self.tr("(s)(m)G"), dict(ha="center", va="center", fontsize="small")),

            ((0.03, 0.125), self.tr("(vs)(g)M"), dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.35), self.tr("(vs)gM"), dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.6), self.tr("(vs)mG"), dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.85), self.tr("(vs)(m)G"), dict(ha="center", va="center", fontsize="small", rotation=60)),

            ((0.845, 0.125), self.tr("(vm)(g)S"), dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.62, 0.35), self.tr("(vm)gS"), dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.37, 0.6), self.tr("(vm)sG"), dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.12, 0.85), self.tr("(vm)(s)G"), dict(ha="center", va="center", fontsize="small", rotation=-60)),

            ((0.05+4*s, -2*s), self.tr("(vs)M"), dict(ha="center", va="top", fontsize="small", rotation=-45)),
            ((0.14+4*s, -2*s), self.tr("(s)M"), dict(ha="center", va="top", fontsize="small")),
            ((0.34+4*s, -2*s), self.tr("sM"), dict(ha="center", va="top", fontsize="small")),
            ((0.64+4*s, -2*s), self.tr("mS"), dict(ha="center", va="top", fontsize="small")),
            ((0.84+4*s, -2*s), self.tr("(m)S"), dict(ha="center", va="top", fontsize="small")),
            ((0.94+4*s, -2*s), self.tr("(vm)S"), dict(ha="center", va="top", fontsize="small", rotation=-45)),

            ((-2*s, 0.03+2*s), self.tr("(vg)M"), dict(ha="right", va="center", fontsize="small")),
            ((-2*s, 0.13+2*s), self.tr("(g)M"), dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.33+2*s), self.tr("gM"), dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.63+2*s), self.tr("mG"), dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.83+2*s), self.tr("(m)G"), dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.97), self.tr("(vm)G"), dict(ha="right", va="center", fontsize="small")),

            ((0.97+0.2*s, 0.03+2*s), self.tr("(vg)S"), dict(ha="left", va="center", fontsize="small")),
            ((0.87+0.2*s, 0.13+2*s), self.tr("(g)S"), dict(ha="left", va="bottom", fontsize="small")),
            ((0.67+0.2*s, 0.33+2*s), self.tr("gS"), dict(ha="left", va="bottom", fontsize="small")),
            ((0.37+0.2*s, 0.63+2*s), self.tr("sG"), dict(ha="left", va="bottom", fontsize="small")),
            ((0.17+0.2*s, 0.83+2*s), self.tr("(s)G"), dict(ha="left", va="bottom", fontsize="small")),
            ((0.03+0.2*s, 0.97+2*s), self.tr("(vs)G"), dict(ha="left", va="bottom", fontsize="small")),

            ((0.03+0.5*s, 0.97+2*s), self.tr("(vs)(vm)G"), dict(ha="left", va="top", fontsize="small")),
            ((-2*s, 0.03+4*s), self.tr("(vg)(vs)G"), dict(ha="right", va="bottom", fontsize="small")),
            ((0.94+0.4*s, 0.03+4*s), self.tr("(vg)(vm)G"), dict(ha="left", va="bottom", fontsize="small"))]

        return LABELS

    def plot_legend(self):
        x0, y0 = 0.7, 0.9
        w, h = 0.5, 0.16
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0+w, x0+w, x0, x0], [y0, y0, y0-h, y0-h, y0], c="black", linewidth=0.8)
        self.axes.text(x0+w/2, y0-span, self.tr("Conventions"), ha="center", va="top", fontsize="small")
        texts = [self.tr("UPPER CASE - Largest component (noun)"),
                 self.tr("Lower case - Descriptive term (adjective)"),
                 self.tr("( ) - Slightly (qualification)"),
                 self.tr("(v ) - Very slightly (qualification)")]
        for row, text in enumerate(texts, 1):
            x = x0+2*span
            y = y0-span - row*(row_h+span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize="x-small")

        x0, y0 = 0.9, 0.7
        w, h = 0.3, 0.32
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0+w, x0+w, x0, x0], [y0, y0, y0-h, y0-h, y0], c="black", linewidth=0.8)
        self.axes.text(x0+w/2, y0-span, self.tr("Term used"), ha="center", va="top", fontsize="small")
        texts = [self.tr("G - Gravel    g - Gravelly"),
                 self.tr("S - Sand      s - Sandy"),
                 self.tr("M - Mud       m - Muddy"),
                 self.tr("(g)  - Slightly gravelly"),
                 self.tr("(s)  - Slightly sandy"),
                 self.tr("(m)  - Slightly muddy"),
                 self.tr("(vg) - Very slightly gravelly"),
                 self.tr("(vs) - Very slightly sandy"),
                 self.tr("(vm) - Very slightly muddy")]
        for row, text in enumerate(texts, 1):
            x = x0+2*span
            y = y0-span - row*(row_h+span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize="x-small")

    def trans_pos(self, a, b):
        a, b = np.array(a), np.array(b)
        y = 0.5 * np.sqrt(3) * b
        x = 0.5*b + a
        return x, y

    def convert_samples(self, samples: typing.List[GrainSizeSample]):
        sand = []
        gravel = []
        for i, sample in enumerate(samples):
            gravel_i, sand_i, mud_i = get_GSM_proportion(sample.classes_φ, sample.distribution)
            sand.append(sand_i)
            gravel.append(gravel_i)
        return sand, gravel

class BP12SSCDiagramChart(DiagramChart):
    def __init__(self, parent=None, toolbar=False):
        super().__init__(parent=parent, toolbar=toolbar)
        self.setWindowTitle(self.tr("Sand-silt-clay diagram (Blott & Pye, 2012)"))

    @property
    def title(self):
        return self.tr("Sand-silt-clay diagram (Blott & Pye, 2012)")

    @property
    def lines(self):
        span = 0.01
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c="black", linewidth=0.8)), # the 3 sides of this equilateral triangle
            # Sand
            ([0.0, 0.99], [0.01, 0.01], dict(c="black", linewidth=0.8)), # sand = Trace
            ([0.0, 0.95], [0.05, 0.05], dict(c="black", linewidth=0.8)), # sand = 5%
            ([0.0, 0.8], [0.2, 0.2], dict(c="black", linewidth=0.8)), # sand = 20%
            ([0.0, 1/3, 0.5], [0.5, 1/3, 0.5], dict(c="black", linewidth=0.8)), # sand = 50%, 33%
            # Silt
            ([0.01, 0.01], [0.0, 0.99], dict(c="black", linewidth=0.8)), # silt = Trace
            ([0.05, 0.05], [0.0, 0.95], dict(c="black", linewidth=0.8)), # silt = 5%
            ([0.2, 0.2], [0.0, 0.8], dict(c="black", linewidth=0.8)), # silt = 20%
            ([0.5, 1/3, 0.5], [0.0, 1/3, 0.5], dict(c="black", linewidth=0.8)), # silt = 50%, 33%
            # Clay
            ([0.99, 0.0], [0.0, 0.99], dict(c="black", linewidth=0.8)), # clay = Trace
            ([0.95, 0.0], [0.0, 0.95], dict(c="black", linewidth=0.8)), # clay = 5%
            ([0.8, 0.0], [0.0, 0.8], dict(c="black", linewidth=0.8)), # clay = 20%
            ([0.5, 1/3, 0.0], [0.0, 1/3, 0.5], dict(c="black", linewidth=0.8))] # clay = 50%, 33%
        ADDITIONAL_LINES = [
            ([0.0, -span], [0.0, -span], dict(c="black", linewidth=0.4)), # S
            ([0.0, -2*span], [1.0, 1+4*span], dict(c="black", linewidth=0.4)), # SI
            ([1.0, 1+2*span], [0.0, -span], dict(c="black", linewidth=0.4)), # C
            # ticks of Sand
            ([0.0, -span], [1.0, 1.0+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.9, 0.9+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.8, 0.8+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.7, 0.7+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.6, 0.6+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.5, 0.5+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.4, 0.4+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.3, 0.3+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.2, 0.2+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.1, 0.1+span], dict(c="black", linewidth=0.4)),
            ([0.0, -span], [0.0, span], dict(c="black", linewidth=0.4)),
            # ticks of Silt
            ([0.0, 0.0+0.1*span], [1.0, 1.0+span], dict(c="black", linewidth=0.4)),
            ([0.1, 0.1+0.1*span], [0.9, 0.9+span], dict(c="black", linewidth=0.4)),
            ([0.2, 0.2+0.1*span], [0.8, 0.8+span], dict(c="black", linewidth=0.4)),
            ([0.3, 0.3+0.1*span], [0.7, 0.7+span], dict(c="black", linewidth=0.4)),
            ([0.4, 0.4+0.1*span], [0.6, 0.6+span], dict(c="black", linewidth=0.4)),
            ([0.5, 0.5+0.1*span], [0.5, 0.5+span], dict(c="black", linewidth=0.4)),
            ([0.6, 0.6+0.1*span], [0.4, 0.4+span], dict(c="black", linewidth=0.4)),
            ([0.7, 0.7+0.1*span], [0.3, 0.3+span], dict(c="black", linewidth=0.4)),
            ([0.8, 0.8+0.1*span], [0.2, 0.2+span], dict(c="black", linewidth=0.4)),
            ([0.9, 0.9+0.1*span], [0.1, 0.1+span], dict(c="black", linewidth=0.4)),
            ([1.0, 1.0+0.1*span], [0.0, span], dict(c="black", linewidth=0.4)),
            # ticks of Caly
            ([0.0, span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.1, 0.1+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.2, 0.2+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.3, 0.3+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.4, 0.4+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.5, 0.5+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.6, 0.6+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.7, 0.7+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.8, 0.8+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([0.9, 0.9+span], [0.0, -span], dict(c="black", linewidth=0.4)),
            ([1.0, 1.0-span], [0.0, -span], dict(c="black", linewidth=0.4)),

            # guide lines
            ([0.03, 0.03+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.13, 0.13+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.33, 0.33+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.63, 0.63+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.83, 0.83+4*span], [0.0, -2*span], dict(c="black", linewidth=0.4)),
            ([0.97, 0.97], [0.0, -2*span], dict(c="black", linewidth=0.4)),

            ([0.0, -2*span], [0.03, 0.03+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.13, 0.13+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.33, 0.33+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.63, 0.63+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.83, 0.83+2*span], dict(c="black", linewidth=0.4)),
            ([0.0, -2*span], [0.97, 0.97], dict(c="black", linewidth=0.4)),

            ([0.97, 0.97+0.2*span], [0.03, 0.03+2*span], dict(c="black", linewidth=0.4)),
            ([0.87, 0.87+0.2*span], [0.13, 0.13+2*span], dict(c="black", linewidth=0.4)),
            ([0.67, 0.67+0.2*span], [0.33, 0.33+2*span], dict(c="black", linewidth=0.4)),
            ([0.37, 0.37+0.2*span], [0.63, 0.63+2*span], dict(c="black", linewidth=0.4)),
            ([0.17, 0.17+0.2*span], [0.83, 0.83+2*span], dict(c="black", linewidth=0.4)),
            ([0.03, 0.03+0.2*span], [0.97, 0.97+2*span], dict(c="black", linewidth=0.4)),

            ([0.03, 0.03+2*span], [0.95, 0.95+2*span], dict(c="black", linewidth=0.4)),
            ([0.03, -2*span], [0.03, 0.03+4*span], dict(c="black", linewidth=0.4)),
            ([0.94, 0.94+0.4*span], [0.03, 0.03+4*span], dict(c="black", linewidth=0.4)),

            ]
        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), self.tr("C"), dict(ha="right", va="top", fontweight="bold")),
            ((-2*s, 1+4*s), self.tr("S"), dict(ha="center", va="bottom", fontweight="bold")),
            ((1+2*s, -s), self.tr("SI"), dict(ha="left", va="top", fontweight="bold")),
            ((-5*s, 0.55), self.tr("Sand %"), dict(ha="right", va="center", fontweight="bold")),
            ((0.5, 0.55), self.tr("Silt %"), dict(ha="left", va="center", fontweight="bold")),
            ((0.5+5*s, -5*np.sqrt(3)*s), self.tr("Clay %"), dict(ha="center", va="top", fontweight="bold")),
            # tick labels of Sand
            ((-s, 1.0+s), self.tr("100%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.9+s), self.tr("90%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.8+s), self.tr("80%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.7+s), self.tr("70%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.6+s), self.tr("60%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.5+s), self.tr("50%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.4+s), self.tr("40%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.3+s), self.tr("30%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.2+s), self.tr("20%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.1+s), self.tr("10%"), dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.0+s), self.tr("0%"), dict(ha="right", va="bottom", fontsize="x-small")),
            # tick labels of Silt
            ((0.0, 1.0+s), self.tr("0%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.1, 0.9+s), self.tr("10%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.2, 0.8+s), self.tr("20%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.3, 0.7+s), self.tr("30%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.4, 0.6+s), self.tr("40%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.5, 0.5+s), self.tr("50%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.6, 0.4+s), self.tr("60%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.7, 0.3+s), self.tr("70%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.8, 0.2+s), self.tr("80%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.9, 0.1+s), self.tr("90%"), dict(ha="left", va="bottom", fontsize="x-small")),
            ((1.0, 0.0+s), self.tr("100%"), dict(ha="left", va="bottom", fontsize="x-small")),
            # tick labels of Clay
            ((2*s, -s), self.tr("100%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.1+2*s, -s), self.tr("90%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.2+2*s, -s), self.tr("80%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.3+2*s, -s), self.tr("70%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.4+2*s, -s), self.tr("60%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.5+2*s, -s), self.tr("50%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.6+2*s, -s), self.tr("40%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.7+2*s, -s), self.tr("30%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.8+2*s, -s), self.tr("20%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.9+2*s, -s), self.tr("10%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((1.0-s, -s), self.tr("0%"), dict(ha="center", va="top", fontsize="x-small", rotation=-45)),

            ((0.125, 0.025), self.tr("(vs)(si)C"), dict(ha="center", va="center", fontsize="small")),
            ((0.35, 0.025), self.tr("(vs)siC"), dict(ha="center", va="center", fontsize="small")),
            ((0.625, 0.025), self.tr("(vs)cSI"), dict(ha="center", va="center", fontsize="small")),
            ((0.850, 0.025), self.tr("(vs)(c)SI"), dict(ha="center", va="center", fontsize="small")),

            ((0.125, 0.125), self.tr("(s)(si)C"), dict(ha="center", va="center", fontsize="small")),
            ((0.3, 0.125), self.tr("(s)siC"), dict(ha="center", va="center", fontsize="small")),
            ((0.575, 0.125), self.tr("(s)cSI"), dict(ha="center", va="center", fontsize="small")),
            ((0.750, 0.125), self.tr("(s)(c)SI"), dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.35), self.tr("(si)sC"), dict(ha="center", va="center", fontsize="small")),
            ((0.3, 0.3), self.tr("ssiC"), dict(ha="center", va="center", fontsize="small")),
            ((0.4, 0.3), self.tr("scSI"), dict(ha="center", va="center", fontsize="small")),
            ((0.525, 0.35), self.tr("(c)sSI"), dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.55), self.tr("(si)cS"), dict(ha="center", va="center", fontsize="small")),
            ((1/3-(0.45-1/3)/2, 0.45), self.tr("sicS"), dict(ha="center", va="center", fontsize="small")),
            ((0.325, 0.55), self.tr("(c)siS"), dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.75), self.tr("(si)(c)S"), dict(ha="center", va="center", fontsize="small")),

            ((0.03, 0.125), self.tr("(vsi)(s)C"), dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.35), self.tr("(vsi)sC"), dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.6), self.tr("(vsi)cS"), dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.85), self.tr("(vsi)(c)S"), dict(ha="center", va="center", fontsize="small", rotation=60)),

            ((0.845, 0.125), self.tr("(vc)(s)SI"), dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.62, 0.35), self.tr("(vc)sSI"), dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.37, 0.6), self.tr("(vc)siS"), dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.12, 0.85), self.tr("(vc)(si)S"), dict(ha="center", va="center", fontsize="small", rotation=-60)),

            ((0.05+4*s, -2*s), self.tr("(vsi)C"), dict(ha="center", va="top", fontsize="small", rotation=-45)),
            ((0.14+4*s, -2*s), self.tr("(si)C"), dict(ha="center", va="top", fontsize="small")),
            ((0.34+4*s, -2*s), self.tr("siC"), dict(ha="center", va="top", fontsize="small")),
            ((0.64+4*s, -2*s), self.tr("cSI"), dict(ha="center", va="top", fontsize="small")),
            ((0.84+4*s, -2*s), self.tr("(c)SI"), dict(ha="center", va="top", fontsize="small")),
            ((0.94+4*s, -2*s), self.tr("(vc)SI"), dict(ha="center", va="top", fontsize="small", rotation=-45)),

            ((-2*s, 0.03+2*s), self.tr("(vs)C"), dict(ha="right", va="center", fontsize="small")),
            ((-2*s, 0.13+2*s), self.tr("(s)C"), dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.33+2*s), self.tr("sC"), dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.63+2*s), self.tr("cS"), dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.83+2*s), self.tr("(c)S"), dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.97), self.tr("(vc)S"), dict(ha="right", va="center", fontsize="small")),

            ((0.97+0.2*s, 0.03+2*s), self.tr("(vs)SI"), dict(ha="left", va="center", fontsize="small")),
            ((0.87+0.2*s, 0.13+2*s), self.tr("(s)SI"), dict(ha="left", va="bottom", fontsize="small")),
            ((0.67+0.2*s, 0.33+2*s), self.tr("sSI"), dict(ha="left", va="bottom", fontsize="small")),
            ((0.37+0.2*s, 0.63+2*s), self.tr("siS"), dict(ha="left", va="bottom", fontsize="small")),
            ((0.17+0.2*s, 0.83+2*s), self.tr("(si)S"), dict(ha="left", va="bottom", fontsize="small")),
            ((0.03+0.2*s, 0.97+2*s), self.tr("(vsi)S"), dict(ha="left", va="bottom", fontsize="small")),

            ((0.03+0.5*s, 0.97+2*s), self.tr("(vsi)(vc)S"), dict(ha="left", va="top", fontsize="small")),
            ((-2*s, 0.03+4*s), self.tr("(vs)(vsi)C"), dict(ha="right", va="bottom", fontsize="small")),
            ((0.94+0.4*s, 0.03+4*s), self.tr("(vs)(vc)SI"), dict(ha="left", va="bottom", fontsize="small"))]

        return LABELS

    def trans_pos(self, a, b):
        a, b = np.array(a), np.array(b)
        y = 0.5 * np.sqrt(3) * b
        x = 0.5*b + a
        return x, y

    def plot_legend(self):
        x0, y0 = 0.7, 0.9
        w, h = 0.5, 0.16
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0+w, x0+w, x0, x0], [y0, y0, y0-h, y0-h, y0], c="black", linewidth=0.8)
        self.axes.text(x0+w/2, y0-span, self.tr("Conventions"), ha="center", va="top", fontsize="small")
        texts = [self.tr("UPPER CASE - Largest component (noun)"),
                 self.tr("Lower case - Descriptive term (adjective)"),
                 self.tr("( ) - Slightly (qualification)"),
                 self.tr("(v ) - Very slightly (qualification)")]
        for row, text in enumerate(texts, 1):
            x = x0+2*span
            y = y0-span - row*(row_h+span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize="x-small")

        x0, y0 = 0.9, 0.7
        w, h = 0.3, 0.32
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0+w, x0+w, x0, x0], [y0, y0, y0-h, y0-h, y0], c="black", linewidth=0.8)
        self.axes.text(x0+w/2, y0-span, self.tr("Term used"), ha="center", va="top", fontsize="small")
        texts = [self.tr("S  - Sand     s  - Sandy"),
                 self.tr("SI - Silt     si - Silty"),
                 self.tr("C  - Clay     c  - Clayey"),
                 self.tr("(s)   - Slightly sandy"),
                 self.tr("(si)  - Slightly silty"),
                 self.tr("(c)   - Slightly clayey"),
                 self.tr("(vs)  - Very slightly sandy"),
                 self.tr("(vsi) - Very slightly silty"),
                 self.tr("(vc)  - Very slightly clayey")]
        for row, text in enumerate(texts, 1):
            x = x0+2*span
            y = y0-span - row*(row_h+span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize="x-small")

    def convert_samples(self, samples: typing.List[GrainSizeSample]):
        silt = []
        sand = []
        for i, sample in enumerate(samples):
            sand_i, silt_i, clay_i = get_SSC_proportion(sample.classes_φ, sample.distribution)
            silt.append(silt_i)
            sand.append(sand_i)

        return silt, sand



if __name__ == "__main__":
    import sys
    from QGrain.entry import setup_app
    app = setup_app()
    chart = BP12SSCDiagramChart(toolbar=True)

    silt = np.random.random(100)
    clay = np.random.random(100)
    sand = np.random.random(100)
    ssc = np.array([sand, silt, clay])
    ssc = ssc / np.sum(ssc, axis=0)

    x, y = chart.trans_pos(ssc[1], ssc[0])
    chart.axes.scatter(x, y, c="black", s=4)

    chart.show()
    sys.exit(app.exec_())
