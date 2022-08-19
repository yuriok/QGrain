__all__ = [
    "DiagramChart",
    "Folk54GSMDiagramChart",
    "Folk54SSCDiagramChart",
    "BP12GSMDiagramChart",
    "BP12SSCDiagramChart",
    "CMDiagramChart"]

from typing import *

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Circle, PathPatch

from . import BaseChart
from . import normal_color, background_color
from ..models import Sample
from ..statistics import cm, proportions_gsm, proportions_ssc, to_phi, to_microns


class DiagramChart(BaseChart):
    def __init__(self, parent=None, figsize=(4.4, 3.1)):
        super().__init__(parent=parent, figsize=figsize)
        self.axes = self._figure.subplots()
        self.draw_base()
        self._sample_batches = []

    def trans_pos(self, a, b) -> Tuple[float, float]:
        pass

    @property
    def title(self):
        return ""

    @property
    def lines(self):
        return []

    @property
    def labels(self):
        return []

    @property
    def n_batches(self) -> int:
        return len(self._sample_batches)

    def plot_legend(self):
        pass

    def draw_base(self):
        self.axes.axis("off")
        self.axes.set_aspect(1)
        # self.axes.set_title(self.title)

        for a, b, kwargs in self.lines:
            x, y = self.trans_pos(a, b)
            self.axes.plot(x, y, **kwargs, label="_")

        for (a, b), text, kwargs in self.labels:
            x, y = self.trans_pos(a, b)
            self.axes.text(x, y, text, color=normal_color(), label="_", **kwargs)

        self.plot_legend()
        self._figure.tight_layout()

    def convert_samples(self, samples: List[Sample]) -> Tuple[Sequence[float], Sequence[float]]:
        pass

    def show_samples(self, samples: List[Sample],
                     append=False, c="#ffffff00",
                     marker=".", ms=8, mfc="red", mew=0.0,
                     **kwargs):
        if len(samples) == 0:
            return
        if not append:
            self.axes.clear()
            self.draw_base()
            self._sample_batches.clear()
        a, b = self.convert_samples(samples)
        x, y = self.trans_pos(a, b)
        self.axes.plot(x, y, c=c, marker=marker, ms=ms, mfc=mfc, mew=mew, label=f"batch_{self.n_batches}",
                       zorder=100, **kwargs)
        self._canvas.draw()
        plot_kwargs = dict(c=c, marker=marker, ms=ms, mfc=mfc, mew=mew)
        plot_kwargs.update(kwargs)
        self._sample_batches.append((samples, plot_kwargs))

    def update_chart(self):
        self._figure.clear()
        self.axes = self._figure.subplots()
        self.draw_base()
        for i, (samples, plot_kwargs) in enumerate(self._sample_batches):
            a, b = self.convert_samples(samples)
            x, y = self.trans_pos(a, b)
            self.axes.plot(x, y, label=f"batch_{i}", **plot_kwargs)
        self._canvas.draw()


class Folk54GSMDiagramChart(DiagramChart):
    def __init__(self, parent=None, figsize=(6, 4.5)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("GSM Diagram (Folk, 1954)"))

    @property
    def title(self):
        return "Gravel-sand-mud diagram (Folk, 1954)"

    @property
    def lines(self):
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c=normal_color(), linewidth=0.8)),
            # the 3 sides of this equilateral triangle
            ([0.0, 1.0], [0.01, 0.01], dict(c=normal_color(), linewidth=0.8)),  # gravel = Trace
            ([0.0, 1.0], [0.05, 0.05], dict(c=normal_color(), linewidth=0.8)),  # gravel = 5%
            ([0.0, 1.0], [0.3, 0.3], dict(c=normal_color(), linewidth=0.8)),  # gravel = 30%
            ([0.0, 1.0], [0.8, 0.8], dict(c=normal_color(), linewidth=0.8)),  # gravel = 80%

            ([1 / 10, 1 / 10], [0.0, 0.05], dict(c=normal_color(), linewidth=0.8)),  # sand: mud = 1:9
            ([1 / 2, 1 / 2], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)),  # sand: mud = 1:1
            ([9 / 10, 9 / 10], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8))]  # sand: mud = 9:1
        ADDITIONAL_LINES = [
            ([0.05, -0.03], [0.02, 0.12], dict(c=normal_color(), linewidth=0.8)),
            ([0.3, 0.3], [0.02, 0.06], dict(c=normal_color(), linewidth=0.8)),
            ([0.7, 0.7], [0.02, 0.06], dict(c=normal_color(), linewidth=0.8)),
            ([0.95, 1.03], [0.02, 0.12], dict(c=normal_color(), linewidth=0.8)),
            ([0.95, 1.03], [0.02, 0.12], dict(c=normal_color(), linewidth=0.8)),
            ([0.95, 1.03], [0.2, 0.275], dict(c=normal_color(), linewidth=0.8)),
            ([0.95, 1.03], [0.5, 0.55], dict(c=normal_color(), linewidth=0.8))]

        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [((-s, -s), "Mud", dict(ha="right", va="top", fontsize=8, fontweight="bold")),
                  ((0.5, 1 + s), "Gravel", dict(ha="center", va="bottom", fontsize=8, fontweight="bold")),
                  ((1 + s, -s), "Sand", dict(ha="left", va="top", fontsize=8, fontweight="bold")),
                  ((-5 * s, 0.55), "Gravel %", dict(ha="right", va="center", fontsize=8, fontweight="bold")),
                  ((0.5, -5 * s), "Sand:Mud Ratio", dict(ha="center", va="top", fontsize=8, fontweight="bold")),

                  ((0.05, -s), "Mud", dict(ha="center", va="top", fontsize=7)),
                  ((0.3, -s), "Sandy Mud", dict(ha="center", va="top", fontsize=7)),
                  ((0.7, -s), "Muddy Sand", dict(ha="center", va="top", fontsize=7)),
                  ((0.95, -s), "Sand", dict(ha="center", va="top", fontsize=7)),

                  ((-0.03, 0.12), "Slightly\nGravelly\nMud", dict(ha="right", va="bottom", fontsize=7)),
                  ((0.3, 0.08), "Slightly Gravelly Sandy Mud", dict(ha="center", va="bottom", fontsize=7)),
                  ((0.7, 0.08), "Slightly Gravelly Muddy Sand", dict(ha="center", va="bottom", fontsize=7)),
                  ((1.03, 0.12), "Slightly\nGravelly\nSand", dict(ha="left", va="bottom", fontsize=7)),

                  ((0.2, 0.2), "Gravelly Mud", dict(ha="center", va="center", fontsize=7)),
                  ((0.7, 0.2), "Gravelly Muddy Sand", dict(ha="center", va="center", fontsize=7)),
                  ((1.03, 0.28), "Gravelly\nSand", dict(ha="left", va="bottom", fontsize=7)),
                  ((0.2, 0.55), "Muddy\nGravel", dict(ha="center", va="center", fontsize=7)),
                  ((0.7, 0.55), "Muddy\nSandy\nGravel", dict(ha="center", va="center", fontsize=7)),
                  ((1.03, 0.55), "Sandy\nGravel", dict(ha="left", va="bottom", fontsize=7)),
                  ((0.5, 0.85), "Gravel", dict(ha="center", va="bottom", fontsize=7)),

                  ((-1 * s, 0.01), "Trace", dict(ha="right", va="center", fontsize=7)),
                  ((-1 * s, 0.05), "5%", dict(ha="right", va="center", fontsize=7)),
                  ((-3 * s, 0.3), "30%", dict(ha="right", va="center", fontsize=7)),
                  ((-8 * s, 0.8), "80%", dict(ha="right", va="center", fontsize=7)),

                  ((1 / 10, -s), "1:9", dict(ha="center", va="top", fontsize=7)),
                  ((1 / 2, -s), "1:1", dict(ha="center", va="top", fontsize=7)),
                  ((9 / 10, -s), "9:1", dict(ha="center", va="top", fontsize=7))]

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
        a[key] = 0.5 * np.sqrt(3) / (0.5 - sand[key])
        b[key] = -a[key] * sand[key]
        x[key] = (y[key] - b[key]) / a[key]
        assert not np.any(np.isnan(x))
        return x, y

    def convert_samples(self, samples: List[Sample]) -> Tuple[Sequence[float], Sequence[float]]:
        sand = []
        gravel = []
        for i, sample in enumerate(samples):
            gravel_i, sand_i, mud_i = proportions_gsm(sample.classes_phi, sample.distribution)
            sand.append(sand_i)
            gravel.append(gravel_i)
        return sand, gravel

    def retranslate(self):
        self.setWindowTitle(self.tr("GSM Diagram (Folk, 1954)"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))


class Folk54SSCDiagramChart(DiagramChart):
    def __init__(self, parent=None, figsize=(6, 4.5)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("SSC Diagram (Folk, 1954)"))

    @property
    def title(self):
        return "Sand-silt-clay diagram (Folk, 1954)"

    @property
    def lines(self):
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c=normal_color(), linewidth=0.8)),
            # the 3 sides of this equilateral triangle
            ([0.0, 1.0], [0.1, 0.1], dict(c=normal_color(), linewidth=0.8)),  # sand = 10%
            ([0.0, 1.0], [0.5, 0.5], dict(c=normal_color(), linewidth=0.8)),  # sand = 50%
            ([0.0, 1.0], [0.9, 0.9], dict(c=normal_color(), linewidth=0.8)),  # sand = 90%
            ([1 / 3, 1 / 3], [0.0, 0.9], dict(c=normal_color(), linewidth=0.8)),  # clay: silt = 1:2
            ([2 / 3, 2 / 3], [0.0, 0.9], dict(c=normal_color(), linewidth=0.8))]  # clay: silt = 2:1
        ADDITIONAL_LINES = []

        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), "Clay", dict(ha="right", va="top", fontsize=8, fontweight="bold")),
            ((0.5, 1 + s), "Sand", dict(ha="center", va="bottom", fontsize=8, fontweight="bold")),
            ((1 + s, -s), "Silt", dict(ha="left", va="top", fontsize=8, fontweight="bold")),
            ((-5 * s, 0.55), "Sand %", dict(ha="right", va="center", fontsize=8, fontweight="bold")),
            ((0.5, -3 * s), "Silt:Clay Ratio", dict(ha="center", va="top", fontsize=8, fontweight="bold")),

            ((0.16, 0.05), "Clay", dict(ha="center", va="center", fontsize=7)),
            ((0.5, 0.05), "Mud", dict(ha="center", va="center", fontsize=7)),
            ((0.84, 0.05), "Silt", dict(ha="center", va="center", fontsize=7)),
            ((0.16, 0.3), "Sandy Clay", dict(ha="center", va="center", fontsize=7)),
            ((0.5, 0.3), "Sandy Mud", dict(ha="center", va="center", fontsize=7)),
            ((0.84, 0.3), "Sandy Silt", dict(ha="center", va="center", fontsize=7)),
            ((0.16, 0.55), "Clayey Sand", dict(ha="center", va="center", fontsize=7)),
            ((0.5, 0.55), "Muddy Sand", dict(ha="center", va="center", fontsize=7)),
            ((0.84, 0.55), "Silty Sand", dict(ha="center", va="center", fontsize=7)),
            ((0.5, 0.91), "Sand", dict(ha="center", va="bottom", fontsize=7)),

            ((-1 * s, 0.1), "10%", dict(ha="right", va="center", fontsize=7)),
            ((-5 * s, 0.5), "50%", dict(ha="right", va="center", fontsize=7)),
            ((-9 * s, 0.9), "90%", dict(ha="right", va="center", fontsize=7)),

            ((1 / 3, -s), "1:2", dict(ha="center", va="top", fontsize=7)),
            ((2 / 3, -s), "2:1", dict(ha="center", va="top", fontsize=7))]

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
        a[key] = 0.5 * np.sqrt(3) / (0.5 - silt[key])
        b[key] = -a[key] * silt[key]
        x[key] = (y[key] - b[key]) / a[key]
        assert not np.any(np.isnan(x))
        return x, y

    def convert_samples(self, samples: List[Sample]) -> Tuple[Sequence[float], Sequence[float]]:
        silt = []
        sand = []
        for i, sample in enumerate(samples):
            sand_i, silt_i, clay_i = proportions_ssc(sample.classes_phi, sample.distribution)
            silt.append(silt_i)
            sand.append(sand_i)
        return silt, sand

    def retranslate(self):
        self.setWindowTitle(self.tr("SSC Diagram (Folk, 1954)"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))


class BP12GSMDiagramChart(DiagramChart):
    def __init__(self, parent=None, figsize=(6, 4.5)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("GSM Diagram (Blott & Pye, 2012)"))

    @property
    def title(self):
        return "Gravel-sand-mud diagram (Blott & Pye, 2012)"

    @property
    def lines(self):
        span = 0.01
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c=normal_color(), linewidth=0.8)),
            # the 3 sides of this equilateral triangle
            # Gravel
            ([0.0, 0.99], [0.01, 0.01], dict(c=normal_color(), linewidth=0.8)),  # gravel = Trace
            ([0.0, 0.95], [0.05, 0.05], dict(c=normal_color(), linewidth=0.8)),  # gravel = 5%
            ([0.0, 0.8], [0.2, 0.2], dict(c=normal_color(), linewidth=0.8)),  # gravel = 20%
            ([0.0, 1 / 3, 0.5], [0.5, 1 / 3, 0.5], dict(c=normal_color(), linewidth=0.8)),  # gravel = 50%, 33%
            # Sand
            ([0.01, 0.01], [0.0, 0.99], dict(c=normal_color(), linewidth=0.8)),  # sand = Trace
            ([0.05, 0.05], [0.0, 0.95], dict(c=normal_color(), linewidth=0.8)),  # sand = 5%
            ([0.2, 0.2], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)),  # sand = 20%
            ([0.5, 1 / 3, 0.5], [0.0, 1 / 3, 0.5], dict(c=normal_color(), linewidth=0.8)),  # sand = 50%, 33%
            # Mud
            ([0.99, 0.0], [0.0, 0.99], dict(c=normal_color(), linewidth=0.8)),  # mud = Trace
            ([0.95, 0.0], [0.0, 0.95], dict(c=normal_color(), linewidth=0.8)),  # mud = 5%
            ([0.8, 0.0], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)),  # mud = 20%
            ([0.5, 1 / 3, 0.0], [0.0, 1 / 3, 0.5], dict(c=normal_color(), linewidth=0.8))]  # mud = 50%, 33%
        ADDITIONAL_LINES = [
            ([0.0, -span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),  # M
            ([0.0, -2 * span], [1.0, 1 + 4 * span], dict(c=normal_color(), linewidth=0.4)),  # G
            ([1.0, 1 + 2 * span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),  # S
            # ticks of Gravel
            ([0.0, -span], [1.0, 1.0 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.9, 0.9 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.8, 0.8 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.7, 0.7 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.6, 0.6 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.5, 0.5 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.4, 0.4 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.3, 0.3 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.2, 0.2 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.1, 0.1 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -span], [0.0, span], dict(c=normal_color(), linewidth=0.4)),
            # ticks of Sand
            ([0.0, 0.0 + 0.1 * span], [1.0, 1.0 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.1, 0.1 + 0.1 * span], [0.9, 0.9 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.2, 0.2 + 0.1 * span], [0.8, 0.8 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.3, 0.3 + 0.1 * span], [0.7, 0.7 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.4, 0.4 + 0.1 * span], [0.6, 0.6 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.5, 0.5 + 0.1 * span], [0.5, 0.5 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.6, 0.6 + 0.1 * span], [0.4, 0.4 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.7, 0.7 + 0.1 * span], [0.3, 0.3 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.8, 0.8 + 0.1 * span], [0.2, 0.2 + span], dict(c=normal_color(), linewidth=0.4)),
            ([0.9, 0.9 + 0.1 * span], [0.1, 0.1 + span], dict(c=normal_color(), linewidth=0.4)),
            ([1.0, 1.0 + 0.1 * span], [0.0, span], dict(c=normal_color(), linewidth=0.4)),
            # ticks of Mud
            ([0.0, span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.1, 0.1 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.2, 0.2 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.3, 0.3 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.4, 0.4 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.5, 0.5 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.6, 0.6 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.7, 0.7 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.8, 0.8 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([0.9, 0.9 + span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),
            ([1.0, 1.0 - span], [0.0, -span], dict(c=normal_color(), linewidth=0.4)),

            # guide lines
            ([0.03, 0.03 + 4 * span], [0.0, -2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.13, 0.13 + 4 * span], [0.0, -2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.33, 0.33 + 4 * span], [0.0, -2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.63, 0.63 + 4 * span], [0.0, -2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.83, 0.83 + 4 * span], [0.0, -2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.97, 0.97], [0.0, -2 * span], dict(c=normal_color(), linewidth=0.4)),

            ([0.0, -2 * span], [0.03, 0.03 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * span], [0.13, 0.13 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * span], [0.33, 0.33 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * span], [0.63, 0.63 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * span], [0.83, 0.83 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * span], [0.97, 0.97], dict(c=normal_color(), linewidth=0.4)),

            ([0.97, 0.97 + 0.2 * span], [0.03, 0.03 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.87, 0.87 + 0.2 * span], [0.13, 0.13 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.67, 0.67 + 0.2 * span], [0.33, 0.33 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.37, 0.37 + 0.2 * span], [0.63, 0.63 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.17, 0.17 + 0.2 * span], [0.83, 0.83 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.03, 0.03 + 0.2 * span], [0.97, 0.97 + 2 * span], dict(c=normal_color(), linewidth=0.4)),

            ([0.03, 0.03 + 2 * span], [0.95, 0.95 + 2 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.03, -2 * span], [0.03, 0.03 + 4 * span], dict(c=normal_color(), linewidth=0.4)),
            ([0.94, 0.94 + 0.4 * span], [0.03, 0.03 + 4 * span], dict(c=normal_color(), linewidth=0.4)),

        ]
        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), "M", dict(ha="right", va="top", fontsize=8, fontweight="bold")),
            ((-2 * s, 1 + 4 * s), "G", dict(ha="center", va="bottom", fontsize=8, fontweight="bold")),
            ((1 + 2 * s, -s), "S", dict(ha="left", va="top", fontsize=8, fontweight="bold")),
            ((-5 * s, 0.55), "Gravel %", dict(ha="right", va="center", fontsize=8, fontweight="bold")),
            ((0.5, 0.55), "Sand %", dict(ha="left", va="center", fontsize=8, fontweight="bold")),
            ((0.5 + 5 * s, -5 * np.sqrt(3) * s), "Mud %", dict(ha="center", va="top", fontsize=8, fontweight="bold")),
            # tick labels of Gravel
            ((-s, 1.0 + s), "100%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.9 + s), "90%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.8 + s), "80%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.7 + s), "70%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.6 + s), "60%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.5 + s), "50%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.4 + s), "40%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.3 + s), "30%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.2 + s), "20%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.1 + s), "10%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.0 + s), "0%", dict(ha="right", va="bottom", fontsize=7)),
            # tick labels of Sand
            ((0.0, 1.0 + s), "0%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.1, 0.9 + s), "10%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.2, 0.8 + s), "20%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.3, 0.7 + s), "30%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.4, 0.6 + s), "40%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.5, 0.5 + s), "50%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.6, 0.4 + s), "60%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.7, 0.3 + s), "70%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.8, 0.2 + s), "80%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.9, 0.1 + s), "90%", dict(ha="left", va="bottom", fontsize=7)),
            ((1.0, 0.0 + s), "100%", dict(ha="left", va="bottom", fontsize=7)),
            # tick labels of Mud
            ((2 * s, -s), "100%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.1 + 2 * s, -s), "90%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.2 + 2 * s, -s), "80%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.3 + 2 * s, -s), "70%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.4 + 2 * s, -s), "60%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.5 + 2 * s, -s), "50%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.6 + 2 * s, -s), "40%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.7 + 2 * s, -s), "30%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.8 + 2 * s, -s), "20%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.9 + 2 * s, -s), "10%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((1.0 - s, -s), "0%", dict(ha="center", va="top", fontsize=7, rotation=-45)),

            ((0.125, 0.025), "(vg)(s)M", dict(ha="center", va="center", fontsize=6)),
            ((0.35, 0.025), "(vg)sM", dict(ha="center", va="center", fontsize=6)),
            ((0.625, 0.025), "(vg)mS", dict(ha="center", va="center", fontsize=6)),
            ((0.850, 0.025), "(vg)(m)S", dict(ha="center", va="center", fontsize=6)),
            ((0.125, 0.125), "(g)(s)M", dict(ha="center", va="center", fontsize=6)),
            ((0.3, 0.125), "(g)sM", dict(ha="center", va="center", fontsize=6)),
            ((0.575, 0.125), "(g)mS", dict(ha="center", va="center", fontsize=6)),
            ((0.750, 0.125), "(g)(m)S", dict(ha="center", va="center", fontsize=6)),
            ((0.125, 0.35), "(s)gM", dict(ha="center", va="center", fontsize=6)),
            ((0.3, 0.3), "gsM", dict(ha="center", va="center", fontsize=6)),
            ((0.4, 0.3), "gmS", dict(ha="center", va="center", fontsize=6)),
            ((0.525, 0.35), "(m)gS", dict(ha="center", va="center", fontsize=6)),
            ((0.125, 0.55), "(s)mG", dict(ha="center", va="center", fontsize=6)),
            ((1 / 3 - (0.45 - 1 / 3) / 2, 0.45), "smG", dict(ha="center", va="center", fontsize=6)),
            ((0.325, 0.55), "(m)sG", dict(ha="center", va="center", fontsize=6)),
            ((0.125, 0.75), "(s)(m)G", dict(ha="center", va="center", fontsize=6)),

            ((0.03, 0.125), "(vs)(g)M", dict(ha="center", va="center", fontsize=6, rotation=60)),
            ((0.03, 0.35), "(vs)gM", dict(ha="center", va="center", fontsize=6, rotation=60)),
            ((0.03, 0.6), "(vs)mG", dict(ha="center", va="center", fontsize=6, rotation=60)),
            ((0.03, 0.85), "(vs)(m)G", dict(ha="center", va="center", fontsize=6, rotation=60)),

            ((0.845, 0.125), "(vm)(g)S", dict(ha="center", va="center", fontsize=6, rotation=-60)),
            ((0.62, 0.35), "(vm)gS", dict(ha="center", va="center", fontsize=6, rotation=-60)),
            ((0.37, 0.6), "(vm)sG", dict(ha="center", va="center", fontsize=6, rotation=-60)),
            ((0.12, 0.85), "(vm)(s)G", dict(ha="center", va="center", fontsize=6, rotation=-60)),

            ((0.05 + 4 * s, -2 * s), "(vs)M", dict(ha="center", va="top", fontsize=6, rotation=-45)),
            ((0.14 + 4 * s, -2 * s), "(s)M", dict(ha="center", va="top", fontsize=6)),
            ((0.34 + 4 * s, -2 * s), "sM", dict(ha="center", va="top", fontsize=6)),
            ((0.64 + 4 * s, -2 * s), "mS", dict(ha="center", va="top", fontsize=6)),
            ((0.84 + 4 * s, -2 * s), "(m)S", dict(ha="center", va="top", fontsize=6)),
            ((0.94 + 4 * s, -2 * s), "(vm)S", dict(ha="center", va="top", fontsize=6, rotation=-45)),

            ((-2 * s, 0.03 + 2 * s), "(vg)M", dict(ha="right", va="center", fontsize=6)),
            ((-2 * s, 0.13 + 2 * s), "(g)M", dict(ha="right", va="bottom", fontsize=6)),
            ((-2 * s, 0.33 + 2 * s), "gM", dict(ha="right", va="bottom", fontsize=6)),
            ((-2 * s, 0.63 + 2 * s), "mG", dict(ha="right", va="bottom", fontsize=6)),
            ((-2 * s, 0.83 + 2 * s), "(m)G", dict(ha="right", va="bottom", fontsize=6)),
            ((-2 * s, 0.97), "(vm)G", dict(ha="right", va="center", fontsize=6)),

            ((0.97 + 0.2 * s, 0.03 + 2 * s), "(vg)S", dict(ha="left", va="center", fontsize=6)),
            ((0.87 + 0.2 * s, 0.13 + 2 * s), "(g)S", dict(ha="left", va="bottom", fontsize=6)),
            ((0.67 + 0.2 * s, 0.33 + 2 * s), "gS", dict(ha="left", va="bottom", fontsize=6)),
            ((0.37 + 0.2 * s, 0.63 + 2 * s), "sG", dict(ha="left", va="bottom", fontsize=6)),
            ((0.17 + 0.2 * s, 0.83 + 2 * s), "(s)G", dict(ha="left", va="bottom", fontsize=6)),
            ((0.03 + 0.2 * s, 0.97 + 2 * s), "(vs)G", dict(ha="left", va="bottom", fontsize=6)),

            ((0.03 + 0.5 * s, 0.97 + 2 * s), "(vs)(vm)G", dict(ha="left", va="top", fontsize=6)),
            ((-2 * s, 0.03 + 4 * s), "(vg)(vs)G", dict(ha="right", va="bottom", fontsize=6)),
            ((0.94 + 0.4 * s, 0.03 + 4 * s), "(vg)(vm)G", dict(ha="left", va="bottom", fontsize=6))]

        return LABELS

    def plot_legend(self):
        x0, y0 = 0.7, 0.9
        w, h = 0.5, 0.16
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 - h, y0 - h, y0], c=normal_color(), linewidth=0.8)
        self.axes.text(x0 + w / 2, y0 - span, "Conventions", ha="center", va="top", fontsize=6,
                       color=normal_color())
        texts = ["UPPER CASE - Largest component (noun)",
                 "Lower case - Descriptive term (adjective)",
                 "( ) - Slightly (qualification)",
                 "(v ) - Very slightly (qualification)"]
        for row, text in enumerate(texts, 1):
            x = x0 + span
            y = y0 - span - row * (row_h + span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize=6, color=normal_color())

        x0, y0 = 0.9, 0.7
        w, h = 0.3, 0.32
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 - h, y0 - h, y0], c=normal_color(), linewidth=0.8)
        self.axes.text(x0 + w / 2, y0 - span, "Term used", ha="center", va="top", fontsize=6,
                       color=normal_color())
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
            x = x0 + span
            y = y0 - span - row * (row_h + span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize=6, color=normal_color())

    def trans_pos(self, a, b):
        a, b = np.array(a), np.array(b)
        y = 0.5 * np.sqrt(3) * b
        x = 0.5 * b + a
        return x, y

    def convert_samples(self, samples: List[Sample]) -> Tuple[Sequence[float], Sequence[float]]:
        sand = []
        gravel = []
        for i, sample in enumerate(samples):
            gravel_i, sand_i, mud_i = proportions_gsm(sample.classes_phi, sample.distribution)
            sand.append(sand_i)
            gravel.append(gravel_i)
        return sand, gravel

    def retranslate(self):
        self.setWindowTitle(self.tr("GSM Diagram (Blott & Pye, 2012)"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))


class BP12SSCDiagramChart(DiagramChart):
    def __init__(self, parent=None, figsize=(6, 4.5)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("SSC Diagram (Blott & Pye, 2012)"))

    @property
    def title(self):
        return "Sand-silt-clay diagram (Blott & Pye, 2012)"

    @property
    def lines(self):
        s = 0.01
        STRUCTURAL_LINES = [
            ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], dict(c=normal_color(), linewidth=0.8)),
            # the 3 sides of this equilateral triangle
            # Sand
            ([0.0, 0.99], [0.01, 0.01], dict(c=normal_color(), linewidth=0.8)),  # sand = Trace
            ([0.0, 0.95], [0.05, 0.05], dict(c=normal_color(), linewidth=0.8)),  # sand = 5%
            ([0.0, 0.8], [0.2, 0.2], dict(c=normal_color(), linewidth=0.8)),  # sand = 20%
            ([0.0, 1 / 3, 0.5], [0.5, 1 / 3, 0.5], dict(c=normal_color(), linewidth=0.8)),  # sand = 50%, 33%
            # Silt
            ([0.01, 0.01], [0.0, 0.99], dict(c=normal_color(), linewidth=0.8)),  # silt = Trace
            ([0.05, 0.05], [0.0, 0.95], dict(c=normal_color(), linewidth=0.8)),  # silt = 5%
            ([0.2, 0.2], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)),  # silt = 20%
            ([0.5, 1 / 3, 0.5], [0.0, 1 / 3, 0.5], dict(c=normal_color(), linewidth=0.8)),  # silt = 50%, 33%
            # Clay
            ([0.99, 0.0], [0.0, 0.99], dict(c=normal_color(), linewidth=0.8)),  # clay = Trace
            ([0.95, 0.0], [0.0, 0.95], dict(c=normal_color(), linewidth=0.8)),  # clay = 5%
            ([0.8, 0.0], [0.0, 0.8], dict(c=normal_color(), linewidth=0.8)),  # clay = 20%
            ([0.5, 1 / 3, 0.0], [0.0, 1 / 3, 0.5], dict(c=normal_color(), linewidth=0.8))]  # clay = 50%, 33%
        ADDITIONAL_LINES = [
            ([0.0, -s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),  # S
            ([0.0, -2 * s], [1.0, 1 + 4 * s], dict(c=normal_color(), linewidth=0.4)),  # SI
            ([1.0, 1 + 2 * s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),  # C
            # ticks of Sand
            ([0.0, -s], [1.0, 1.0 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.9, 0.9 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.8, 0.8 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.7, 0.7 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.6, 0.6 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.5, 0.5 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.4, 0.4 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.3, 0.3 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.2, 0.2 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.1, 0.1 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -s], [0.0, s], dict(c=normal_color(), linewidth=0.4)),
            # ticks of Silt
            ([0.0, 0.0 + 0.1 * s], [1.0, 1.0 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.1, 0.1 + 0.1 * s], [0.9, 0.9 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.2, 0.2 + 0.1 * s], [0.8, 0.8 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.3, 0.3 + 0.1 * s], [0.7, 0.7 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.4, 0.4 + 0.1 * s], [0.6, 0.6 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.5, 0.5 + 0.1 * s], [0.5, 0.5 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.6, 0.6 + 0.1 * s], [0.4, 0.4 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.7, 0.7 + 0.1 * s], [0.3, 0.3 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.8, 0.8 + 0.1 * s], [0.2, 0.2 + s], dict(c=normal_color(), linewidth=0.4)),
            ([0.9, 0.9 + 0.1 * s], [0.1, 0.1 + s], dict(c=normal_color(), linewidth=0.4)),
            ([1.0, 1.0 + 0.1 * s], [0.0, s], dict(c=normal_color(), linewidth=0.4)),
            # ticks of Caly
            ([0.0, s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.1, 0.1 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.2, 0.2 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.3, 0.3 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.4, 0.4 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.5, 0.5 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.6, 0.6 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.7, 0.7 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.8, 0.8 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([0.9, 0.9 + s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),
            ([1.0, 1.0 - s], [0.0, -s], dict(c=normal_color(), linewidth=0.4)),

            # guide lines
            ([0.03, 0.03 + 4 * s], [0.0, -2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.13, 0.13 + 4 * s], [0.0, -2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.33, 0.33 + 4 * s], [0.0, -2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.63, 0.63 + 4 * s], [0.0, -2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.83, 0.83 + 4 * s], [0.0, -2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.97, 0.97], [0.0, -2 * s], dict(c=normal_color(), linewidth=0.4)),

            ([0.0, -2 * s], [0.03, 0.03 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * s], [0.13, 0.13 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * s], [0.33, 0.33 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * s], [0.63, 0.63 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * s], [0.83, 0.83 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.0, -2 * s], [0.97, 0.97], dict(c=normal_color(), linewidth=0.4)),

            ([0.97, 0.97 + 0.2 * s], [0.03, 0.03 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.87, 0.87 + 0.2 * s], [0.13, 0.13 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.67, 0.67 + 0.2 * s], [0.33, 0.33 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.37, 0.37 + 0.2 * s], [0.63, 0.63 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.17, 0.17 + 0.2 * s], [0.83, 0.83 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.03, 0.03 + 0.2 * s], [0.97, 0.97 + 2 * s], dict(c=normal_color(), linewidth=0.4)),

            ([0.03, 0.03 + 2 * s], [0.95, 0.95 + 2 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.03, -2 * s], [0.03, 0.03 + 4 * s], dict(c=normal_color(), linewidth=0.4)),
            ([0.94, 0.94 + 0.4 * s], [0.03, 0.03 + 4 * s], dict(c=normal_color(), linewidth=0.4)),

        ]
        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.01
        LABELS = [
            ((-s, -s), "C", dict(ha="right", va="top", fontsize=8, fontweight="bold")),
            ((-2 * s, 1 + 4 * s), "S", dict(ha="center", va="bottom", fontsize=8, fontweight="bold")),
            ((1 + 2 * s, -s), "SI", dict(ha="left", va="top", fontsize=8, fontweight="bold")),
            ((-5 * s, 0.55), "Sand %", dict(ha="right", va="center", fontsize=8, fontweight="bold")),
            ((0.5, 0.55), "Silt %", dict(ha="left", va="center", fontsize=8, fontweight="bold")),
            ((0.5 + 5 * s, -5 * np.sqrt(3) * s), "Clay %", dict(ha="center", va="top", fontsize=8, fontweight="bold")),
            # tick labels of Sand
            ((-s, 1.0 + s), "100%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.9 + s), "90%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.8 + s), "80%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.7 + s), "70%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.6 + s), "60%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.5 + s), "50%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.4 + s), "40%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.3 + s), "30%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.2 + s), "20%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.1 + s), "10%", dict(ha="right", va="bottom", fontsize=7)),
            ((-s, 0.0 + s), "0%", dict(ha="right", va="bottom", fontsize=7)),
            # tick labels of Silt
            ((0.0, 1.0 + s), "0%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.1, 0.9 + s), "10%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.2, 0.8 + s), "20%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.3, 0.7 + s), "30%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.4, 0.6 + s), "40%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.5, 0.5 + s), "50%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.6, 0.4 + s), "60%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.7, 0.3 + s), "70%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.8, 0.2 + s), "80%", dict(ha="left", va="bottom", fontsize=7)),
            ((0.9, 0.1 + s), "90%", dict(ha="left", va="bottom", fontsize=7)),
            ((1.0, 0.0 + s), "100%", dict(ha="left", va="bottom", fontsize=7)),
            # tick labels of Clay
            ((2 * s, -s), "100%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.1 + 2 * s, -s), "90%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.2 + 2 * s, -s), "80%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.3 + 2 * s, -s), "70%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.4 + 2 * s, -s), "60%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.5 + 2 * s, -s), "50%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.6 + 2 * s, -s), "40%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.7 + 2 * s, -s), "30%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.8 + 2 * s, -s), "20%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((0.9 + 2 * s, -s), "10%", dict(ha="center", va="top", fontsize=7, rotation=-45)),
            ((1.0 - s, -s), "0%", dict(ha="center", va="top", fontsize=7, rotation=-45)),

            ((0.125, 0.025), "(vs)(si)C", dict(ha="center", va="center", fontsize=6)),
            ((0.35, 0.025), "(vs)siC", dict(ha="center", va="center", fontsize=6)),
            ((0.625, 0.025), "(vs)cSI", dict(ha="center", va="center", fontsize=6)),
            ((0.850, 0.025), "(vs)(c)SI", dict(ha="center", va="center", fontsize=6)),

            ((0.125, 0.125), "(s)(si)C", dict(ha="center", va="center", fontsize=6)),
            ((0.3, 0.125), "(s)siC", dict(ha="center", va="center", fontsize=6)),
            ((0.575, 0.125), "(s)cSI", dict(ha="center", va="center", fontsize=6)),
            ((0.750, 0.125), "(s)(c)SI", dict(ha="center", va="center", fontsize=6)),
            ((0.125, 0.35), "(si)sC", dict(ha="center", va="center", fontsize=6)),
            ((0.3, 0.3), "ssiC", dict(ha="center", va="center", fontsize=6)),
            ((0.4, 0.3), "scSI", dict(ha="center", va="center", fontsize=6)),
            ((0.525, 0.35), "(c)sSI", dict(ha="center", va="center", fontsize=6)),
            ((0.125, 0.55), "(si)cS", dict(ha="center", va="center", fontsize=6)),
            ((1 / 3 - (0.45 - 1 / 3) / 2, 0.45), "sicS", dict(ha="center", va="center", fontsize=6)),
            ((0.325, 0.55), "(c)siS", dict(ha="center", va="center", fontsize=6)),
            ((0.125, 0.75), "(si)(c)S", dict(ha="center", va="center", fontsize=6)),

            ((0.03, 0.125), "(vsi)(s)C", dict(ha="center", va="center", fontsize=6, rotation=60)),
            ((0.03, 0.35), "(vsi)sC", dict(ha="center", va="center", fontsize=6, rotation=60)),
            ((0.03, 0.6), "(vsi)cS", dict(ha="center", va="center", fontsize=6, rotation=60)),
            ((0.03, 0.85), "(vsi)(c)S", dict(ha="center", va="center", fontsize=6, rotation=60)),

            ((0.845, 0.125), "(vc)(s)SI", dict(ha="center", va="center", fontsize=6, rotation=-60)),
            ((0.62, 0.35), "(vc)sSI", dict(ha="center", va="center", fontsize=6, rotation=-60)),
            ((0.37, 0.6), "(vc)siS", dict(ha="center", va="center", fontsize=6, rotation=-60)),
            ((0.12, 0.85), "(vc)(si)S", dict(ha="center", va="center", fontsize=6, rotation=-60)),

            ((0.05 + 4 * s, -2 * s), "(vsi)C", dict(ha="center", va="top", fontsize=6, rotation=-45)),
            ((0.14 + 4 * s, -2 * s), "(si)C", dict(ha="center", va="top", fontsize=6)),
            ((0.34 + 4 * s, -2 * s), "siC", dict(ha="center", va="top", fontsize=6)),
            ((0.64 + 4 * s, -2 * s), "cSI", dict(ha="center", va="top", fontsize=6)),
            ((0.84 + 4 * s, -2 * s), "(c)SI", dict(ha="center", va="top", fontsize=6)),
            ((0.94 + 4 * s, -2 * s), "(vc)SI", dict(ha="center", va="top", fontsize=6, rotation=-45)),

            ((-2 * s, 0.03 + 2 * s), "(vs)C", dict(ha="right", va="center", fontsize=6)),
            ((-2 * s, 0.13 + 2 * s), "(s)C", dict(ha="right", va="bottom", fontsize=6)),
            ((-2 * s, 0.33 + 2 * s), "sC", dict(ha="right", va="bottom", fontsize=6)),
            ((-2 * s, 0.63 + 2 * s), "cS", dict(ha="right", va="bottom", fontsize=6)),
            ((-2 * s, 0.83 + 2 * s), "(c)S", dict(ha="right", va="bottom", fontsize=6)),
            ((-2 * s, 0.97), "(vc)S", dict(ha="right", va="center", fontsize=6)),

            ((0.97 + 0.2 * s, 0.03 + 2 * s), "(vs)SI", dict(ha="left", va="center", fontsize=6)),
            ((0.87 + 0.2 * s, 0.13 + 2 * s), "(s)SI", dict(ha="left", va="bottom", fontsize=6)),
            ((0.67 + 0.2 * s, 0.33 + 2 * s), "sSI", dict(ha="left", va="bottom", fontsize=6)),
            ((0.37 + 0.2 * s, 0.63 + 2 * s), "siS", dict(ha="left", va="bottom", fontsize=6)),
            ((0.17 + 0.2 * s, 0.83 + 2 * s), "(si)S", dict(ha="left", va="bottom", fontsize=6)),
            ((0.03 + 0.2 * s, 0.97 + 2 * s), "(vsi)S", dict(ha="left", va="bottom", fontsize=6)),

            ((0.03 + 0.5 * s, 0.97 + 2 * s), "(vsi)(vc)S", dict(ha="left", va="top", fontsize=6)),
            ((-2 * s, 0.03 + 4 * s), "(vs)(vsi)C", dict(ha="right", va="bottom", fontsize=6)),
            ((0.94 + 0.4 * s, 0.03 + 4 * s), "(vs)(vc)SI", dict(ha="left", va="bottom", fontsize=6))]

        return LABELS

    def trans_pos(self, a, b):
        a, b = np.array(a), np.array(b)
        y = 0.5 * np.sqrt(3) * b
        x = 0.5 * b + a
        return x, y

    def plot_legend(self):
        x0, y0 = 0.7, 0.9
        w, h = 0.5, 0.16
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 - h, y0 - h, y0], c=normal_color(), linewidth=0.8)
        self.axes.text(x0 + w / 2, y0 - span, "Conventions", ha="center", va="top", fontsize=6,
                       color=normal_color())
        texts = ["UPPER CASE - Largest component (noun)",
                 "Lower case - Descriptive term (adjective)",
                 "( ) - Slightly (qualification)",
                 "(v ) - Very slightly (qualification)"]
        for row, text in enumerate(texts, 1):
            x = x0 + span
            y = y0 - span - row * (row_h + span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize=6, color=normal_color())

        x0, y0 = 0.9, 0.7
        w, h = 0.3, 0.32
        span = 0.01
        row_h = 0.02
        self.axes.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 - h, y0 - h, y0], c=normal_color(), linewidth=0.8)
        self.axes.text(x0 + w / 2, y0 - span, "Term used", ha="center", va="top", fontsize=6,
                       color=normal_color())
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
            x = x0 + span
            y = y0 - span - row * (row_h + span)
            self.axes.text(x, y, text, ha="left", va="top", fontsize=6, color=normal_color())

    def convert_samples(self, samples: List[Sample]) -> Tuple[Sequence[float], Sequence[float]]:
        silt = []
        sand = []
        for i, sample in enumerate(samples):
            sand_i, silt_i, clay_i = proportions_ssc(sample.classes_phi, sample.distribution)
            silt.append(silt_i)
            sand.append(sand_i)

        return silt, sand

    def retranslate(self):
        self.setWindowTitle(self.tr("SSC Diagram (Blott & Pye, 2012)"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))


class CMDiagramChart(DiagramChart):
    """
    The C-M diagram referred to Mycielska-Dowgiałło and Ludwikowska-Kędzia (2011).

    Mycielska-Dowgiałło, E., Ludwikowska-Kędzia, M., 2011. Alternative interpretations of grain-size data from
        Quaternary deposits. Geologos 17. https://doi.org/10.2478/v10118-011-0010-9

    """
    GRID_C = (0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0)
    GRID_M = (0.004, 0.01, 0.015, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8)

    def __init__(self, parent=None, figsize=(4.4, 6.0)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("C-M Diagram"))

    @property
    def title(self) -> str:
        return "C-M Diagram"

    @property
    def lines(self):
        STRUCTURAL_LINES = [([0.003, 0.01, 1.0, 1.0, 0.003, 0.003], [0.01, 0.01, 1.0, 60.0, 60.0, 0.01],
                             dict(c=normal_color(), linewidth=0.8))]
        for m in self.GRID_M:
            line = ([m, m], [max(m, 0.01), 60.0], dict(c=normal_color(), linewidth=0.5))
            STRUCTURAL_LINES.append(line)
        for c in self.GRID_C:
            line = ([0.003, min(c, 1.0)], [c, c], dict(c=normal_color(), linewidth=0.5))
            STRUCTURAL_LINES.append(line)

        ADDITIONAL_LINES = [([0.365, 1.25], [0.600, 0.600], dict(c=normal_color(), linestyle="--", linewidth=0.8)),
                            ([0.50, 1.25], [0.900, 0.900], dict(c=normal_color(), linestyle="--", linewidth=0.8)),
                            ([0.50, 1.25], [18.00, 18.00], dict(c=normal_color(), linestyle="--", linewidth=0.8)),
                            ([0.15, 1.25], [0.2383, 0.2383], dict(c=normal_color(), linestyle="--", linewidth=0.8)),
                            ([0.27, 1.25], [0.4366, 0.4366], dict(c=normal_color(), linestyle="--", linewidth=0.8)),
                            ([0.50, 1.25], [1.0734, 1.0734], dict(c=normal_color(), linestyle="--", linewidth=0.8)),

                            ([0.7, 0.7], [2.05, 1.5], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.24, 0.275], [2.00, 1.2], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.15, 0.2], [1.0, 0.9], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.35, 0.15], [0.32, 0.32], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.07, 0.07], [0.16, 0.2], dict(c=normal_color(), linestyle="-", linewidth=0.5)),

                            ([0.006, 0.006], [0.065, 0.05], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.5, 0.5], [40, 25], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.25, 0.35], [5.0, 5.0], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.58, 0.30], [0.5, 0.8], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.02, 0.025], [0.95, 0.5], dict(c=normal_color(), linestyle="-", linewidth=0.5)),
                            ([0.01, 0.02], [0.15, 0.1], dict(c=normal_color(), linestyle="-", linewidth=0.5))]
        for x, y, kwargs in ADDITIONAL_LINES:
            kwargs["zorder"] = 10

        return STRUCTURAL_LINES + ADDITIONAL_LINES

    @property
    def labels(self):
        s = 0.05
        LABELS = [
            ((0.00125, 2.0), "C (first percentile)",
             dict(ha="right", va="center", rotation=90.0, fontsize=8, fontweight="bold")),
            ((0.05, 110.0), "M (median)", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.15, 0.15), "C = M", dict(ha="center", va="top", rotation=45.0, fontsize=8, fontweight="bold")),
            ((0.003 * (1 - s), 60.0), "mm", dict(ha="right", va="center", fontsize=6)),
            ((0.002 * (1 - s), 60.0), r"phi ($\phi$)", dict(ha="right", va="center", fontsize=6)),
            ((1.0, 60.0 * (1 + s)), "1.0", dict(ha="center", va="bottom", fontsize=6)),
            ((1.0, 70.0 * (1 + s)), "0.0", dict(ha="center", va="bottom", fontsize=6)),
            ((1.5, 60.0 * (1 + s)), "mm", dict(ha="center", va="bottom", fontsize=6)),
            ((1.5, 70.0 * (1 + s)), r"phi ($\phi$)", dict(ha="center", va="bottom", fontsize=6)),

            ((0.70, 3.30), "I", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.15, 3.30), "II", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.03, 3.30), "III", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.51, 0.68), "IV", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.15, 0.68), "V", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.05, 0.13), "VI", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.03, 0.68), "VII", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.007, 0.68), "VIII", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.007, 3.30), "IX", dict(ha="center", va="center", fontsize=8, fontweight="bold")),

            ((0.0175, 0.200), "S", dict(ha="center", va="center", fontsize=7)),
            ((0.1168, 0.234), "R", dict(ha="center", va="center", fontsize=7)),
            ((0.2077, 0.436), "Q", dict(ha="center", va="center", fontsize=7)),
            ((0.2200, 1.142), "P", dict(ha="center", va="center", fontsize=7)),
            ((0.5500, 1.265), "O", dict(ha="center", va="center", fontsize=7)),
            ((0.8000, 1.482), "N", dict(ha="center", va="center", fontsize=7)),
            ((0.0055, 0.032), "T", dict(ha="center", va="center", fontsize=7)),

            ((0.0125, 0.475), "S", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.1500, 0.475), "R", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.3500, 0.975), "Q", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.3700, 23.700), "P", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.7000, 23.700), "O", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((0.0250, 0.125), "T", dict(ha="center", va="center", fontsize=8, fontweight="bold")),

            ((1.5, 0.595), "Cu", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((1.5, 0.900), "Cs", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((1.5, 18.00), "Cr", dict(ha="center", va="center", fontsize=8, fontweight="bold")),
            ((1.5, 0.2383), "Cu", dict(ha="center", va="center", fontsize=8, fontstyle="italic")),
            ((1.5, 0.4366), "Cs", dict(ha="center", va="center", fontsize=8, fontstyle="italic")),
            ((1.5, 1.0734), "Cr", dict(ha="center", va="center", fontsize=8, fontstyle="italic")),

            ((0.7, 2.25), "rolling", dict(ha="center", va="center", fontsize=6)),
            ((0.15, 2.0), "rolling and\nsuspension",
             dict(ha="center", va="center", fontsize=6, backgroundcolor=background_color())),
            ((0.10, 1.2), "suspension\nand rolling",
             dict(ha="center", va="center", fontsize=6, backgroundcolor=background_color())),
            ((0.36, 0.32), "graded suspension", dict(ha="left", va="center", fontsize=6)),
            ((0.075, 0.125), "uniform\nsuspension", dict(ha="center", va="center", fontsize=6)),
            ((0.0068, 0.08), "pelagic\nsuspension",
             dict(ha="center", va="center", fontsize=6, backgroundcolor=background_color())),

            ((0.5, 45.0), "rolling", dict(ha="center", va="center", fontsize=6, backgroundcolor=background_color())),
            ((0.25, 4.86), "saltation and rolling",
             dict(ha="right", va="center", fontsize=6, backgroundcolor=background_color())),
            ((0.60, 0.55), "saltation", dict(ha="left", va="top", fontsize=6)),
            ((0.02, 1.0), "suspension", dict(ha="center", va="bottom", fontsize=6, backgroundcolor=background_color())),
            ((0.0068, 0.14), "overbank-pool\nfacies\nsuspension",
             dict(ha="center", va="center", fontsize=6, backgroundcolor=background_color()))]

        for m in self.GRID_M:
            label_mm = ((m, 60.0 * (1 + s)), f"{m}"[1:], dict(ha="center", va="bottom", fontsize=6))
            label_phi = ((m, 70.0 * (1 + s)), f"{to_phi(m * 1000):0.1f}", dict(ha="center", va="bottom", fontsize=6))
            LABELS.append(label_mm)
            LABELS.append(label_phi)
        for c in self.GRID_C:
            label_mm = ((0.003 * (1 - s), c), f"{c}", dict(ha="right", va="center", fontsize=6))
            label_phi = ((0.002 * (1 - s), c), f"{to_phi(c * 1000):0.1f}", dict(ha="right", va="center", fontsize=6))
            LABELS.append(label_mm)
            LABELS.append(label_phi)

        return LABELS

    def trans_pos(self, a, b):
        return np.log10(a), np.log10(b)

    def plot_legend(self):
        def pixel_to_mc(x, y):
            m = 0.01 * 10 ** ((x - 388) / 400)
            c = 20.0 / 10 ** ((y - 345) / 407)
            return m, c

        passega_t = Circle(self.trans_pos(*pixel_to_mc(284, 1484)), radius=(369 - 284) / 400,
                           linewidth=1.0, edgecolor=normal_color(), facecolor="#747474", alpha=0.8)
        self.axes.add_patch(passega_t)
        passega_chanel = dict(data=[
            (Path.MOVETO, (512, 1182)), (Path.LINETO, (781, 1182)), (Path.CURVE3, (802, 1178)),
            (Path.CURVE3, (824, 1159)), (Path.LINETO, (925, 1059)), (Path.CURVE3, (938, 1039)),
            (Path.CURVE3, (940, 1027)), (Path.LINETO, (940, 892)), (Path.CURVE4, (944, 873)),
            (Path.CURVE4, (952, 861)), (Path.CURVE4, (979, 859)), (Path.LINETO, (1092, 859)),
            (Path.CURVE3, (1126, 852)), (Path.CURVE3, (1145, 838)), (Path.LINETO, (1157, 829)),
            (Path.CURVE4, (1198, 786)), (Path.CURVE4, (1165, 747)), (Path.CURVE4, (1120, 793)),
            (Path.CURVE3, (1105, 805)), (Path.CURVE3, (1073, 811)), (Path.LINETO, (948, 811)),
            (Path.CURVE4, (889, 811)), (Path.CURVE4, (889, 863)), (Path.CURVE4, (889, 877)),
            (Path.LINETO, (891, 1007)), (Path.CURVE3, (885, 1029)), (Path.CURVE3, (870, 1043)),
            (Path.LINETO, (806, 1106)), (Path.CURVE3, (781, 1127)), (Path.CURVE3, (751, 1133)),
            (Path.LINETO, (481, 1133)), (Path.CURVE4, (426, 1133)), (Path.CURVE4, (426, 1182)),
            (Path.CURVE4, (481, 1182)), (Path.CLOSEPOLY, (512, 1182))],
            linewidth=1.0, facecolor="#747474", alpha=0.8)
        lk2000_chanel = dict(data=[
            (Path.MOVETO, (512, 1094)), (Path.LINETO, (817, 1094)), (Path.CURVE3, (890, 1094)),
            (Path.CURVE3, (925, 1056)), (Path.LINETO, (1037, 946)), (Path.CURVE3, (1061, 911)),
            (Path.CURVE3, (1067, 859)), (Path.LINETO, (1067, 395)), (Path.CURVE3, (1065, 354)),
            (Path.CURVE3, (1110, 350)), (Path.CURVE4, (1165, 350)), (Path.CURVE4, (1165, 272)),
            (Path.CURVE4, (1110, 272)), (Path.LINETO, (1037, 272)), (Path.CURVE4, (1000, 272)),
            (Path.CURVE4, (985, 287)), (Path.CURVE4, (985, 324)), (Path.LINETO, (985, 821)),
            (Path.CURVE3, (985, 862)), (Path.CURVE3, (964, 893)), (Path.LINETO, (907, 945)),
            (Path.CURVE3, (882, 965)), (Path.CURVE3, (839, 965)), (Path.LINETO, (456, 965)),
            (Path.CURVE4, (363, 965)), (Path.CURVE4, (363, 1094)), (Path.CURVE4, (456, 1094)),
            (Path.CLOSEPOLY, (512, 1094))],
            linewidth=1.0, facecolor="#cfcfcf", alpha=0.8)
        lk2000_t = dict(data=[
            (Path.MOVETO, (512, 1339)), (Path.LINETO, (669, 1183)), (Path.CURVE4, (709, 1143)),
            (Path.CURVE4, (658, 1091)), (Path.CURVE4, (618, 1131)), (Path.LINETO, (444, 1304)),
            (Path.CURVE4, (404, 1344)), (Path.CURVE4, (456, 1397)), (Path.CURVE4, (496, 1357)),
            (Path.CLOSEPOLY, (512, 1339))],
            linewidth=1.0, facecolor="#cfcfcf", alpha=0.8)

        patchs = [lk2000_chanel, lk2000_t, passega_chanel]
        for patch_define in patchs:
            codes, pixels = zip(*patch_define["data"])
            verts = []
            for x, y in pixels:
                verts.append(self.trans_pos(*pixel_to_mc(x, y)))
            path = Path(verts, codes)
            patch = PathPatch(path, linewidth=patch_define["linewidth"], edgecolor=normal_color(),
                              facecolor=patch_define["facecolor"], alpha=patch_define["alpha"])
            self.axes.add_patch(patch)
        # caption = "This C-M diagram is modified after Mycielska-Dowgiałło & Ludwikowska-Kędzia (2011). " \
        #           "See the description of symbols from Passega (1964),
        #           "Passega & Byramjee (1969) and Mycielska-Dowgiałło & Ludwikowska-Kędzia (2011)."
        # self.axes.text(*self.trans_pos(0.003, 0.01), caption, fontsize=6, ha="left", va="top", wrap=True)

    def convert_samples(self, samples: List[Sample]) -> Tuple[Sequence[float], Sequence[float]]:
        c = []
        m = []
        for i, sample in enumerate(samples):
            c_i, m_i = cm(sample.classes_phi, sample.distribution)
            c.append(to_microns(c_i) / 1000)
            m.append(to_microns(m_i) / 1000)
        return m, c

    def retranslate(self):
        self.setWindowTitle(self.tr("C-M Diagram"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.save_figure_action.setText(self.tr("Save Figure"))
