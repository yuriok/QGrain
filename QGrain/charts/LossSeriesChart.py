__all__ = ["LossSeriesChart"]

from typing import *

from . import BaseChart


class LossSeriesChart(BaseChart):
    def __init__(self, parent=None, figsize=(3, 2.5)):
        super().__init__(parent=parent, figsize=figsize)
        self.setWindowTitle(self.tr("Loss Series"))
        self._axes = self._figure.subplots()
        self._last_result = None

    def show_loss_series(self, series: Sequence[float], ylabel: str, title: str = ""):
        self._axes.clear()
        self._axes.plot(series, label="loss_series")
        self._axes.set_xlabel(self.tr("Iteration"))
        self._axes.set_ylabel(ylabel)
        self._axes.set_title(title)
        self._figure.tight_layout()
        self._canvas.draw()
        self._last_result = series, ylabel, title

    def update_chart(self):
        if self._last_result is not None:
            self._figure.clear()
            self._axes = self._figure.subplots()
            self.show_loss_series(*self._last_result)

    def retranslate(self):
        self.setWindowTitle(self.tr("Loss Series"))
        self.edit_figure_action.setText(self.tr("Edit Figure"))
        self.configure_subplots_action.setText(self.tr("Configure Subplots"))
        self.save_figure_action.setText(self.tr("Save Figure"))
