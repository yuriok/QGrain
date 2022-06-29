__all__ = [
    "DiagramChart",
    "Folk54GSMDiagramChart",
    "Folk54SSCDiagramChart",
    "BP12GSMDiagramChart",
    "BP12SSCDiagramChart"]

import typing

import numpy as np

from ..model import GrainSizeSample
from ..statistics import get_GSM_proportion, get_SSC_proportion
from .BaseChart import BaseChart
from .config_matplotlib import normal_color


class DiagramChart(BaseChart):
    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent=parent, figsize=figsize)
        self.axes = self.figure.subplots()
        self.draw_base()
        self.__sample_batchses = []

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
            self.axes.text(x, y, text, color=normal_color(), **kwargs)

        self.plot_legend()
        self.figure.tight_layout()

    def convert_samples(self, samples: typing.List[GrainSizeSample]):
        pass

    def show_samples(self, samples: typing.List[GrainSizeSample],
                     append=False, c="#ffffff00",
                     marker=".", ms=8, mfc="red", mew=0.0,
                     **kwargs):
        if len(samples) == 0:
            return
        if not append:
            self.axes.clear()
            self.draw_base()
            self.__sample_batchses.clear()
        a, b = self.convert_samples(samples)
        x, y = self.trans_pos(a, b)
        self.axes.plot(x, y, c=c, marker=marker, ms=ms, mfc=mfc, mew=mew, **kwargs)
        self.canvas.draw()
        plot_kwargs = dict(c=c, marker=marker, ms=ms, mfc=mfc, mew=mew)
        plot_kwargs.update(kwargs)
        self.__sample_batchses.append((samples, plot_kwargs))

    def update_chart(self):
        self.figure.clear()
        self.axes = self.figure.subplots()
        self.draw_base()
        for samples, plot_kwargs in self.__sample_batchses:
            a, b = self.convert_samples(samples)
            x, y = self.trans_pos(a, b)
            self.axes.plot(x, y, **plot_kwargs)
        self.canvas.draw()


class Folk54GSMDiagramChart(DiagramChart):
    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("GSM Diagram (Folk, 1954)"))

    @property
    def title(self):
        return "Gravel-sand-mud diagram (Folk, 1954)"

    @property
    def lines(self):
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c=normal_color(), linewidth=0.8)), # the 3 sides of this equilateral triangle
            ([0.0, 1.0], [0.01, 0.01], dict(c=normal_color(), linewidth=0.8)), # gravel = Trace
            ([0.0, 1.0], [0.05, 0.05], dict(c=normal_color(), linewidth=0.8)), # gravel = 5%
            ([0.0, 1.0], [0.3, 0.3], dict(c=normal_color(), linewidth=0.8)), # gravel = 30%
            ([0.0, 1.0], [0.8, 0.8], dict(c=normal_color(), linewidth=0.8)), # gravel = 80%

            ([1/10, 1/10], [0.0, 0.05], dict(c=normal_color(), linewidth=0.8)), # sand: mud = 1:9
            ([1/2, 1/2], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)), # sand: mud = 1:1
            ([9/10, 9/10], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8))] # sand: mud = 9:1
        ADDTIONAL_LINES = [
            ([0.05, -0.03], [0.02, 0.12], dict(c=normal_color(), linewidth=0.8)),
            ([0.3, 0.3], [0.02, 0.06], dict(c=normal_color(), linewidth=0.8)),
            ([0.7, 0.7], [0.02, 0.06], dict(c=normal_color(), linewidth=0.8)),
            ([0.95, 1.03], [0.02, 0.12], dict(c=normal_color(), linewidth=0.8)),
            ([0.95, 1.03], [0.02, 0.12], dict(c=normal_color(), linewidth=0.8)),
            ([0.95, 1.03], [0.2, 0.275], dict(c=normal_color(), linewidth=0.8)),
            ([0.95, 1.03], [0.5, 0.55], dict(c=normal_color(), linewidth=0.8))]

        return STRUCTURAL_LINES + ADDTIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [((-s, -s), "Mud", dict(ha="right", va="top", fontweight="bold")),
                  ((0.5, 1+s), "Gravel", dict(ha="center", va="bottom", fontweight="bold")),
                  ((1+s, -s), "Sand", dict(ha="left", va="top", fontweight="bold")),
                  ((-5*s, 0.55), "Gravel %", dict(ha="right", va="center", fontweight="bold")),
                  ((0.5, -5*s), "Sand:Mud Ratio", dict(ha="center", va="top", fontweight="bold")),

                  ((0.05, -s), "Mud", dict(ha="center", va="top", fontsize="x-small")),
                  ((0.3, -s), "Sandy Mud", dict(ha="center", va="top", fontsize="x-small")),
                  ((0.7, -s), "Muddy Sand", dict(ha="center", va="top", fontsize="x-small")),
                  ((0.95, -s), "Sand", dict(ha="center", va="top", fontsize="x-small")),

                  ((-0.03, 0.12), "Slightly\nGravelly\nMud", dict(ha="right", va="bottom", fontsize="x-small")),
                  ((0.3, 0.08), "Slightly Gravelly Sandy Mud", dict(ha="center", va="bottom", fontsize="x-small")),
                  ((0.7, 0.08), "Slightly Gravelly Muddy Sand", dict(ha="center", va="bottom", fontsize="x-small")),
                  ((1.03, 0.12), "Slightly\nGravelly\nSand", dict(ha="left", va="bottom", fontsize="x-small")),

                  ((0.2, 0.2), "Gravelly Mud", dict(ha="center", va="center", fontsize="x-small")),
                  ((0.7, 0.2), "Gravelly Muddy Sand", dict(ha="center", va="center", fontsize="x-small")),
                  ((1.03, 0.28), "Gravelly\nSand", dict(ha="left", va="bottom", fontsize="x-small")),
                  ((0.2, 0.55), "Muddy\nGravel", dict(ha="center", va="center", fontsize="x-small")),
                  ((0.7, 0.55), "Muddy\nSandy\nGravel", dict(ha="center", va="center", fontsize="x-small")),
                  ((1.03, 0.55), "Sandy\nGravel", dict(ha="left", va="bottom", fontsize="x-small")),
                  ((0.5, 0.85), "Gravel", dict(ha="center", va="bottom", fontsize="x-small")),

                  ((-1*s, 0.01), "Trace", dict(ha="right", va="center", fontsize="x-small")),
                  ((-1*s, 0.05), "5%", dict(ha="right", va="center", fontsize="x-small")),
                  ((-3*s, 0.3), "30%", dict(ha="right", va="center", fontsize="x-small")),
                  ((-8*s, 0.8), "80%", dict(ha="right", va="center", fontsize="x-small")),

                  ((1/10, -s), "1:9", dict(ha="center", va="top", fontsize="x-small")),
                  ((1/2, -s), "1:1", dict(ha="center", va="top", fontsize="x-small")),
                  ((9/10, -s), "9:1", dict(ha="center", va="top", fontsize="x-small"))]

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

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("GSM Diagram (Folk, 1954)"))


class Folk54SSCDiagramChart(DiagramChart):
    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("SSC Diagram (Folk, 1954)"))

    @property
    def title(self):
        return "Sand-silt-clay diagram (Folk, 1954)"

    @property
    def lines(self):
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c=normal_color(), linewidth=0.8)), # the 3 sides of this equilateral triangle
            ([0.0, 1.0], [0.1, 0.1], dict(c=normal_color(), linewidth=0.8)), # sand = 10%
            ([0.0, 1.0], [0.5, 0.5], dict(c=normal_color(), linewidth=0.8)), # sand = 50%
            ([0.0, 1.0], [0.9, 0.9], dict(c=normal_color(), linewidth=0.8)), # sand = 90%
            ([1/3, 1/3], [0.0, 0.9], dict(c=normal_color(), linewidth=0.8)), # clay: silt = 1:2
            ([2/3, 2/3], [0.0, 0.9], dict(c=normal_color(), linewidth=0.8))] # clay: silt = 2:1
        ADDTIONAL_LINES = []

        return STRUCTURAL_LINES + ADDTIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), "Clay", dict(ha="right", va="top", fontweight="bold")),
            ((0.5, 1+s), "Sand", dict(ha="center", va="bottom", fontweight="bold")),
            ((1+s, -s), "Silt", dict(ha="left", va="top", fontweight="bold")),
            ((-5*s, 0.55), "Sand %", dict(ha="right", va="center", fontweight="bold")),
            ((0.5, -3*s), "Silt:Clay Ratio", dict(ha="center", va="top", fontweight="bold")),

            ((0.16, 0.05), "Clay", dict(ha="center", va="center", fontsize="x-small")),
            ((0.5, 0.05), "Mud", dict(ha="center", va="center", fontsize="x-small")),
            ((0.84, 0.05), "Silt", dict(ha="center", va="center", fontsize="x-small")),
            ((0.16, 0.3), "Sandy Clay", dict(ha="center", va="center", fontsize="x-small")),
            ((0.5, 0.3), "Sandy Mud", dict(ha="center", va="center", fontsize="x-small")),
            ((0.84, 0.3), "Sandy Silt", dict(ha="center", va="center", fontsize="x-small")),
            ((0.16, 0.55), "Clayey Sand", dict(ha="center", va="center", fontsize="x-small")),
            ((0.5, 0.55), "Muddy Sand", dict(ha="center", va="center", fontsize="x-small")),
            ((0.84, 0.55), "Silty Sand", dict(ha="center", va="center", fontsize="x-small")),
            ((0.5, 0.91), "Sand", dict(ha="center", va="bottom", fontsize="x-small")),

            ((-1*s, 0.1), "10%", dict(ha="right", va="center", fontsize="x-small")),
            ((-5*s, 0.5), "50%", dict(ha="right", va="center", fontsize="x-small")),
            ((-9*s, 0.9), "90%", dict(ha="right", va="center", fontsize="x-small")),

            ((1/3, -s), "1:2", dict(ha="center", va="top", fontsize="x-small")),
            ((2/3, -s), "2:1", dict(ha="center", va="top", fontsize="x-small"))]

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

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("SSC Diagram (Folk, 1954)"))


class BP12GSMDiagramChart(DiagramChart):
    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("GSM Diagram (Blott & Pye, 2012)"))

    @property
    def title(self):
        return "Gravel-sand-mud diagram (Blott & Pye, 2012)"

    @property
    def lines(self):
        span = 0.01
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c=normal_color(), linewidth=0.8)), # the 3 sides of this equilateral triangle
            # Gravel
            ([0.0, 0.99], [0.01, 0.01], dict(c=normal_color(), linewidth=0.8)), # gravel = Trace
            ([0.0, 0.95], [0.05, 0.05], dict(c=normal_color(), linewidth=0.8)), # gravel = 5%
            ([0.0, 0.8], [0.2, 0.2], dict(c=normal_color(), linewidth=0.8)), # gravel = 20%
            ([0.0, 1/3, 0.5], [0.5, 1/3, 0.5], dict(c=normal_color(), linewidth=0.8)), # gravel = 50%, 33%
            # Sand
            ([0.01, 0.01], [0.0, 0.99], dict(c=normal_color(), linewidth=0.8)), # sand = Trace
            ([0.05, 0.05], [0.0, 0.95], dict(c=normal_color(), linewidth=0.8)), # sand = 5%
            ([0.2, 0.2], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)), # sand = 20%
            ([0.5, 1/3, 0.5], [0.0, 1/3, 0.5], dict(c=normal_color(), linewidth=0.8)), # sand = 50%, 33%
            # Mud
            ([0.99, 0.0], [0.0, 0.99], dict(c=normal_color(), linewidth=0.8)), # mud = Trace
            ([0.95, 0.0], [0.0, 0.95], dict(c=normal_color(), linewidth=0.8)), # mud = 5%
            ([0.8, 0.0], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)), # mud = 20%
            ([0.5, 1/3, 0.0], [0.0, 1/3, 0.5], dict(c=normal_color(), linewidth=0.8))] # mud = 50%, 33%
        ADDITIONAL_LINES = [
            ([0.0, -span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)), # M
            ([0.0, -2*span], [1.0, 1+4*span], dict(c=normal_color(), linewidth=0.4)), # G
            ([1.0, 1+2*span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)), # S
            # ticks of Gravel
            ([0.0, -span], [1.0, 1.0+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.9, 0.9+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.8, 0.8+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.7, 0.7+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.6, 0.6+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.5, 0.5+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.4, 0.4+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.3, 0.3+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.2, 0.2+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.1, 0.1+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.0, span], dict(c=normal_color(), linewidth=0.4)),
            # ticks of Sand
            ([0.0, 0.0+0.1*span], [1.0, 1.0+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.1, 0.1+0.1*span], [0.9, 0.9+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.2, 0.2+0.1*span], [0.8, 0.8+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.3, 0.3+0.1*span], [0.7, 0.7+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.4, 0.4+0.1*span], [0.6, 0.6+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.5, 0.5+0.1*span], [0.5, 0.5+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.6, 0.6+0.1*span], [0.4, 0.4+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.7, 0.7+0.1*span], [0.3, 0.3+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.8, 0.8+0.1*span], [0.2, 0.2+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.9, 0.9+0.1*span], [0.1, 0.1+span], dict(c=normal_color(), linewidth=0.4)),
            ([1.0, 1.0+0.1*span], [0.0, span], dict(c=normal_color(), linewidth=0.4)),
            # ticks of Mud
            ([0.0, span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.1, 0.1+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.2, 0.2+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.3, 0.3+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.4, 0.4+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.5, 0.5+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.6, 0.6+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.7, 0.7+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.8, 0.8+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.9, 0.9+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([1.0, 1.0-span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),

            # guide lines
            ([0.03, 0.03+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.13, 0.13+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.33, 0.33+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.63, 0.63+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.83, 0.83+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.97, 0.97], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),

            ([0.0, -2*span], [0.03, 0.03+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.13, 0.13+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.33, 0.33+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.63, 0.63+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.83, 0.83+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.97, 0.97], dict(c=normal_color(), linewidth=0.4)),

            ([0.97, 0.97+0.2*span], [0.03, 0.03+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.87, 0.87+0.2*span], [0.13, 0.13+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.67, 0.67+0.2*span], [0.33, 0.33+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.37, 0.37+0.2*span], [0.63, 0.63+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.17, 0.17+0.2*span], [0.83, 0.83+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.03, 0.03+0.2*span], [0.97, 0.97+2*span], dict(c=normal_color(), linewidth=0.4)),

            ([0.03, 0.03+2*span], [0.95, 0.95+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.03, -2*span], [0.03, 0.03+4*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.94, 0.94+0.4*span], [0.03, 0.03+4*span], dict(c=normal_color(), linewidth=0.4)),

            ]
        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), "M", dict(ha="right", va="top", fontweight="bold")),
            ((-2*s, 1+4*s), "G", dict(ha="center", va="bottom", fontweight="bold")),
            ((1+2*s, -s), "S", dict(ha="left", va="top", fontweight="bold")),
            ((-5*s, 0.55), "Gravel %", dict(ha="right", va="center", fontweight="bold")),
            ((0.5, 0.55), "Sand %", dict(ha="left", va="center", fontweight="bold")),
            ((0.5+5*s, -5*np.sqrt(3)*s), "Mud %", dict(ha="center", va="top", fontweight="bold")),
            # tick labels of Gravel
            ((-s, 1.0+s), "100%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.9+s), "90%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.8+s), "80%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.7+s), "70%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.6+s), "60%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.5+s), "50%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.4+s), "40%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.3+s), "30%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.2+s), "20%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.1+s), "10%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.0+s), "0%", dict(ha="right", va="bottom", fontsize="x-small")),
            # tick labels of Sand
            ((0.0, 1.0+s), "0%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.1, 0.9+s), "10%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.2, 0.8+s), "20%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.3, 0.7+s), "30%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.4, 0.6+s), "40%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.5, 0.5+s), "50%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.6, 0.4+s), "60%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.7, 0.3+s), "70%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.8, 0.2+s), "80%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.9, 0.1+s), "90%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((1.0, 0.0+s), "100%", dict(ha="left", va="bottom", fontsize="x-small")),
            # tick labels of Mud
            ((2*s, -s), "100%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.1+2*s, -s), "90%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.2+2*s, -s), "80%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.3+2*s, -s), "70%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.4+2*s, -s), "60%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.5+2*s, -s), "50%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.6+2*s, -s), "40%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.7+2*s, -s), "30%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.8+2*s, -s), "20%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.9+2*s, -s), "10%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((1.0-s, -s), "0%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),

            ((0.125, 0.025), "(vg)(s)M", dict(ha="center", va="center", fontsize="small")),
            ((0.35, 0.025), "(vg)sM", dict(ha="center", va="center", fontsize="small")),
            ((0.625, 0.025), "(vg)mS", dict(ha="center", va="center", fontsize="small")),
            ((0.850, 0.025), "(vg)(m)S", dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.125), "(g)(s)M", dict(ha="center", va="center", fontsize="small")),
            ((0.3, 0.125), "(g)sM", dict(ha="center", va="center", fontsize="small")),
            ((0.575, 0.125), "(g)mS", dict(ha="center", va="center", fontsize="small")),
            ((0.750, 0.125), "(g)(m)S", dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.35), "(s)gM", dict(ha="center", va="center", fontsize="small")),
            ((0.3, 0.3), "gsM", dict(ha="center", va="center", fontsize="small")),
            ((0.4, 0.3), "gmS", dict(ha="center", va="center", fontsize="small")),
            ((0.525, 0.35), "(m)gS", dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.55), "(s)mG", dict(ha="center", va="center", fontsize="small")),
            ((1/3-(0.45-1/3)/2, 0.45), "smG", dict(ha="center", va="center", fontsize="small")),
            ((0.325, 0.55), "(m)sG", dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.75), "(s)(m)G", dict(ha="center", va="center", fontsize="small")),

            ((0.03, 0.125), "(vs)(g)M", dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.35), "(vs)gM", dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.6), "(vs)mG", dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.85), "(vs)(m)G", dict(ha="center", va="center", fontsize="small", rotation=60)),

            ((0.845, 0.125), "(vm)(g)S", dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.62, 0.35), "(vm)gS", dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.37, 0.6), "(vm)sG", dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.12, 0.85), "(vm)(s)G", dict(ha="center", va="center", fontsize="small", rotation=-60)),

            ((0.05+4*s, -2*s), "(vs)M", dict(ha="center", va="top", fontsize="small", rotation=-45)),
            ((0.14+4*s, -2*s), "(s)M", dict(ha="center", va="top", fontsize="small")),
            ((0.34+4*s, -2*s), "sM", dict(ha="center", va="top", fontsize="small")),
            ((0.64+4*s, -2*s), "mS", dict(ha="center", va="top", fontsize="small")),
            ((0.84+4*s, -2*s), "(m)S", dict(ha="center", va="top", fontsize="small")),
            ((0.94+4*s, -2*s), "(vm)S", dict(ha="center", va="top", fontsize="small", rotation=-45)),

            ((-2*s, 0.03+2*s), "(vg)M", dict(ha="right", va="center", fontsize="small")),
            ((-2*s, 0.13+2*s), "(g)M", dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.33+2*s), "gM", dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.63+2*s), "mG", dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.83+2*s), "(m)G", dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.97), "(vm)G", dict(ha="right", va="center", fontsize="small")),

            ((0.97+0.2*s, 0.03+2*s), "(vg)S", dict(ha="left", va="center", fontsize="small")),
            ((0.87+0.2*s, 0.13+2*s), "(g)S", dict(ha="left", va="bottom", fontsize="small")),
            ((0.67+0.2*s, 0.33+2*s), "gS", dict(ha="left", va="bottom", fontsize="small")),
            ((0.37+0.2*s, 0.63+2*s), "sG", dict(ha="left", va="bottom", fontsize="small")),
            ((0.17+0.2*s, 0.83+2*s), "(s)G", dict(ha="left", va="bottom", fontsize="small")),
            ((0.03+0.2*s, 0.97+2*s), "(vs)G", dict(ha="left", va="bottom", fontsize="small")),

            ((0.03+0.5*s, 0.97+2*s), "(vs)(vm)G", dict(ha="left", va="top", fontsize="small")),
            ((-2*s, 0.03+4*s), "(vg)(vs)G", dict(ha="right", va="bottom", fontsize="small")),
            ((0.94+0.4*s, 0.03+4*s), "(vg)(vm)G", dict(ha="left", va="bottom", fontsize="small"))]

        return LABELS

    def plot_legend(self):
        x0, y0 = 0.7, 0.9
        w, h = 0.5, 0.16
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0+w, x0+w, x0, x0], [y0, y0, y0-h, y0-h, y0], c=normal_color(), linewidth=0.8)
        self.axes.text(x0+w/2, y0-span, "Conventions", ha="center", va="top", fontsize="small", color=normal_color())
        texts = ["UPPER CASE - Largest component (noun)",
                 "Lower case - Descriptive term (adjective)",
                 "( ) - Slightly (qualification)",
                 "(v ) - Very slightly (qualification)"]
        for row, text in enumerate(texts, 1):
            x = x0+2*span
            y = y0-span - row*(row_h+span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize="x-small", color=normal_color())

        x0, y0 = 0.9, 0.7
        w, h = 0.3, 0.32
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0+w, x0+w, x0, x0], [y0, y0, y0-h, y0-h, y0], c=normal_color(), linewidth=0.8)
        self.axes.text(x0+w/2, y0-span, "Term used", ha="center", va="top", fontsize="small", color=normal_color())
        texts = ["G - Gravel    g - Gravelly",
                 "S - Sand      s - Sandy",
                 "M - Mud       m - Muddy",
                 "(g)  - Slightly gravelly",
                 "(s)  - Slightly sandy",
                 "(m)  - Slightly muddy",
                 "(vg) - Very slightly gravelly",
                 "(vs) - Very slightly sandy",
                 "(vm) - Very slightly muddy"]
        for row, text in enumerate(texts, 1):
            x = x0+2*span
            y = y0-span - row*(row_h+span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize="x-small", color=normal_color())

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

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("GSM Diagram (Blott & Pye, 2012)"))


class BP12SSCDiagramChart(DiagramChart):
    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("SSC Diagram (Blott & Pye, 2012)"))

    @property
    def title(self):
        return "Sand-silt-clay diagram (Blott & Pye, 2012)"

    @property
    def lines(self):
        span = 0.01
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c=normal_color(), linewidth=0.8)), # the 3 sides of this equilateral triangle
            # Sand
            ([0.0, 0.99], [0.01, 0.01], dict(c=normal_color(), linewidth=0.8)), # sand = Trace
            ([0.0, 0.95], [0.05, 0.05], dict(c=normal_color(), linewidth=0.8)), # sand = 5%
            ([0.0, 0.8], [0.2, 0.2], dict(c=normal_color(), linewidth=0.8)), # sand = 20%
            ([0.0, 1/3, 0.5], [0.5, 1/3, 0.5], dict(c=normal_color(), linewidth=0.8)), # sand = 50%, 33%
            # Silt
            ([0.01, 0.01], [0.0, 0.99], dict(c=normal_color(), linewidth=0.8)), # silt = Trace
            ([0.05, 0.05], [0.0, 0.95], dict(c=normal_color(), linewidth=0.8)), # silt = 5%
            ([0.2, 0.2], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)), # silt = 20%
            ([0.5, 1/3, 0.5], [0.0, 1/3, 0.5], dict(c=normal_color(), linewidth=0.8)), # silt = 50%, 33%
            # Clay
            ([0.99, 0.0], [0.0, 0.99], dict(c=normal_color(), linewidth=0.8)), # clay = Trace
            ([0.95, 0.0], [0.0, 0.95], dict(c=normal_color(), linewidth=0.8)), # clay = 5%
            ([0.8, 0.0], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)), # clay = 20%
            ([0.5, 1/3, 0.0], [0.0, 1/3, 0.5], dict(c=normal_color(), linewidth=0.8))] # clay = 50%, 33%
        ADDITIONAL_LINES = [
            ([0.0, -span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)), # S
            ([0.0, -2*span], [1.0, 1+4*span], dict(c=normal_color(), linewidth=0.4)), # SI
            ([1.0, 1+2*span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)), # C
            # ticks of Sand
            ([0.0, -span], [1.0, 1.0+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.9, 0.9+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.8, 0.8+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.7, 0.7+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.6, 0.6+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.5, 0.5+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.4, 0.4+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.3, 0.3+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.2, 0.2+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.1, 0.1+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.0, span], dict(c=normal_color(), linewidth=0.4)),
            # ticks of Silt
            ([0.0, 0.0+0.1*span], [1.0, 1.0+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.1, 0.1+0.1*span], [0.9, 0.9+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.2, 0.2+0.1*span], [0.8, 0.8+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.3, 0.3+0.1*span], [0.7, 0.7+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.4, 0.4+0.1*span], [0.6, 0.6+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.5, 0.5+0.1*span], [0.5, 0.5+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.6, 0.6+0.1*span], [0.4, 0.4+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.7, 0.7+0.1*span], [0.3, 0.3+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.8, 0.8+0.1*span], [0.2, 0.2+span], dict(c=normal_color(), linewidth=0.4)),
            ([0.9, 0.9+0.1*span], [0.1, 0.1+span], dict(c=normal_color(), linewidth=0.4)),
            ([1.0, 1.0+0.1*span], [0.0, span], dict(c=normal_color(), linewidth=0.4)),
            # ticks of Caly
            ([0.0, span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.1, 0.1+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.2, 0.2+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.3, 0.3+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.4, 0.4+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.5, 0.5+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.6, 0.6+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.7, 0.7+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.8, 0.8+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.9, 0.9+span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([1.0, 1.0-span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),

            # guide lines
            ([0.03, 0.03+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.13, 0.13+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.33, 0.33+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.63, 0.63+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.83, 0.83+4*span], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.97, 0.97], [0.0, -2*span], dict(c=normal_color(), linewidth=0.4)),

            ([0.0, -2*span], [0.03, 0.03+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.13, 0.13+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.33, 0.33+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.63, 0.63+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.83, 0.83+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2*span], [0.97, 0.97], dict(c=normal_color(), linewidth=0.4)),

            ([0.97, 0.97+0.2*span], [0.03, 0.03+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.87, 0.87+0.2*span], [0.13, 0.13+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.67, 0.67+0.2*span], [0.33, 0.33+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.37, 0.37+0.2*span], [0.63, 0.63+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.17, 0.17+0.2*span], [0.83, 0.83+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.03, 0.03+0.2*span], [0.97, 0.97+2*span], dict(c=normal_color(), linewidth=0.4)),

            ([0.03, 0.03+2*span], [0.95, 0.95+2*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.03, -2*span], [0.03, 0.03+4*span], dict(c=normal_color(), linewidth=0.4)),
            ([0.94, 0.94+0.4*span], [0.03, 0.03+4*span], dict(c=normal_color(), linewidth=0.4)),

            ]
        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), "C", dict(ha="right", va="top", fontweight="bold")),
            ((-2*s, 1+4*s), "S", dict(ha="center", va="bottom", fontweight="bold")),
            ((1+2*s, -s), "SI", dict(ha="left", va="top", fontweight="bold")),
            ((-5*s, 0.55), "Sand %", dict(ha="right", va="center", fontweight="bold")),
            ((0.5, 0.55), "Silt %", dict(ha="left", va="center", fontweight="bold")),
            ((0.5+5*s, -5*np.sqrt(3)*s), "Clay %", dict(ha="center", va="top", fontweight="bold")),
            # tick labels of Sand
            ((-s, 1.0+s), "100%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.9+s), "90%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.8+s), "80%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.7+s), "70%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.6+s), "60%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.5+s), "50%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.4+s), "40%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.3+s), "30%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.2+s), "20%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.1+s), "10%", dict(ha="right", va="bottom", fontsize="x-small")),
            ((-s, 0.0+s), "0%", dict(ha="right", va="bottom", fontsize="x-small")),
            # tick labels of Silt
            ((0.0, 1.0+s), "0%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.1, 0.9+s), "10%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.2, 0.8+s), "20%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.3, 0.7+s), "30%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.4, 0.6+s), "40%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.5, 0.5+s), "50%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.6, 0.4+s), "60%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.7, 0.3+s), "70%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.8, 0.2+s), "80%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((0.9, 0.1+s), "90%", dict(ha="left", va="bottom", fontsize="x-small")),
            ((1.0, 0.0+s), "100%", dict(ha="left", va="bottom", fontsize="x-small")),
            # tick labels of Clay
            ((2*s, -s), "100%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.1+2*s, -s), "90%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.2+2*s, -s), "80%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.3+2*s, -s), "70%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.4+2*s, -s), "60%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.5+2*s, -s), "50%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.6+2*s, -s), "40%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.7+2*s, -s), "30%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.8+2*s, -s), "20%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((0.9+2*s, -s), "10%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),
            ((1.0-s, -s), "0%", dict(ha="center", va="top", fontsize="x-small", rotation=-45)),

            ((0.125, 0.025), "(vs)(si)C", dict(ha="center", va="center", fontsize="small")),
            ((0.35, 0.025), "(vs)siC", dict(ha="center", va="center", fontsize="small")),
            ((0.625, 0.025), "(vs)cSI", dict(ha="center", va="center", fontsize="small")),
            ((0.850, 0.025), "(vs)(c)SI", dict(ha="center", va="center", fontsize="small")),

            ((0.125, 0.125), "(s)(si)C", dict(ha="center", va="center", fontsize="small")),
            ((0.3, 0.125), "(s)siC", dict(ha="center", va="center", fontsize="small")),
            ((0.575, 0.125), "(s)cSI", dict(ha="center", va="center", fontsize="small")),
            ((0.750, 0.125), "(s)(c)SI", dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.35), "(si)sC", dict(ha="center", va="center", fontsize="small")),
            ((0.3, 0.3), "ssiC", dict(ha="center", va="center", fontsize="small")),
            ((0.4, 0.3), "scSI", dict(ha="center", va="center", fontsize="small")),
            ((0.525, 0.35), "(c)sSI", dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.55), "(si)cS", dict(ha="center", va="center", fontsize="small")),
            ((1/3-(0.45-1/3)/2, 0.45), "sicS", dict(ha="center", va="center", fontsize="small")),
            ((0.325, 0.55), "(c)siS", dict(ha="center", va="center", fontsize="small")),
            ((0.125, 0.75), "(si)(c)S", dict(ha="center", va="center", fontsize="small")),

            ((0.03, 0.125), "(vsi)(s)C", dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.35), "(vsi)sC", dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.6), "(vsi)cS", dict(ha="center", va="center", fontsize="small", rotation=60)),
            ((0.03, 0.85), "(vsi)(c)S", dict(ha="center", va="center", fontsize="small", rotation=60)),

            ((0.845, 0.125), "(vc)(s)SI", dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.62, 0.35), "(vc)sSI", dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.37, 0.6), "(vc)siS", dict(ha="center", va="center", fontsize="small", rotation=-60)),
            ((0.12, 0.85), "(vc)(si)S", dict(ha="center", va="center", fontsize="small", rotation=-60)),

            ((0.05+4*s, -2*s), "(vsi)C", dict(ha="center", va="top", fontsize="small", rotation=-45)),
            ((0.14+4*s, -2*s), "(si)C", dict(ha="center", va="top", fontsize="small")),
            ((0.34+4*s, -2*s), "siC", dict(ha="center", va="top", fontsize="small")),
            ((0.64+4*s, -2*s), "cSI", dict(ha="center", va="top", fontsize="small")),
            ((0.84+4*s, -2*s), "(c)SI", dict(ha="center", va="top", fontsize="small")),
            ((0.94+4*s, -2*s), "(vc)SI", dict(ha="center", va="top", fontsize="small", rotation=-45)),

            ((-2*s, 0.03+2*s), "(vs)C", dict(ha="right", va="center", fontsize="small")),
            ((-2*s, 0.13+2*s), "(s)C", dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.33+2*s), "sC", dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.63+2*s), "cS", dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.83+2*s), "(c)S", dict(ha="right", va="bottom", fontsize="small")),
            ((-2*s, 0.97), "(vc)S", dict(ha="right", va="center", fontsize="small")),

            ((0.97+0.2*s, 0.03+2*s), "(vs)SI", dict(ha="left", va="center", fontsize="small")),
            ((0.87+0.2*s, 0.13+2*s), "(s)SI", dict(ha="left", va="bottom", fontsize="small")),
            ((0.67+0.2*s, 0.33+2*s), "sSI", dict(ha="left", va="bottom", fontsize="small")),
            ((0.37+0.2*s, 0.63+2*s), "siS", dict(ha="left", va="bottom", fontsize="small")),
            ((0.17+0.2*s, 0.83+2*s), "(si)S", dict(ha="left", va="bottom", fontsize="small")),
            ((0.03+0.2*s, 0.97+2*s), "(vsi)S", dict(ha="left", va="bottom", fontsize="small")),

            ((0.03+0.5*s, 0.97+2*s), "(vsi)(vc)S", dict(ha="left", va="top", fontsize="small")),
            ((-2*s, 0.03+4*s), "(vs)(vsi)C", dict(ha="right", va="bottom", fontsize="small")),
            ((0.94+0.4*s, 0.03+4*s), "(vs)(vc)SI", dict(ha="left", va="bottom", fontsize="small"))]

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
        self.axes.plot([x0, x0+w, x0+w, x0, x0], [y0, y0, y0-h, y0-h, y0], c=normal_color(), linewidth=0.8)
        self.axes.text(x0+w/2, y0-span, "Conventions", ha="center", va="top", fontsize="small", color=normal_color())
        texts = ["UPPER CASE - Largest component (noun)",
                 "Lower case - Descriptive term (adjective)",
                 "( ) - Slightly (qualification)",
                 "(v ) - Very slightly (qualification)"]
        for row, text in enumerate(texts, 1):
            x = x0+2*span
            y = y0-span - row*(row_h+span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize="x-small", color=normal_color())

        x0, y0 = 0.9, 0.7
        w, h = 0.3, 0.32
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0+w, x0+w, x0, x0], [y0, y0, y0-h, y0-h, y0], c=normal_color(), linewidth=0.8)
        self.axes.text(x0+w/2, y0-span, "Term used", ha="center", va="top", fontsize="small", color=normal_color())
        texts = ["S  - Sand     s  - Sandy",
                 "SI - Silt     si - Silty",
                 "C  - Clay     c  - Clayey",
                 "(s)   - Slightly sandy",
                 "(si)  - Slightly silty",
                 "(c)   - Slightly clayey",
                 "(vs)  - Very slightly sandy",
                 "(vsi) - Very slightly silty",
                 "(vc)  - Very slightly clayey"]
        for row, text in enumerate(texts, 1):
            x = x0+2*span
            y = y0-span - row*(row_h+span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize="x-small", color=normal_color())

    def convert_samples(self, samples: typing.List[GrainSizeSample]):
        silt = []
        sand = []
        for i, sample in enumerate(samples):
            sand_i, silt_i, clay_i = get_SSC_proportion(sample.classes_φ, sample.distribution)
            silt.append(silt_i)
            sand.append(sand_i)

        return silt, sand

    def retranslate(self):
        super().retranslate()
        self.setWindowTitle(self.tr("SSC Diagram (Blott & Pye, 2012)"))
