"""Compact operator plot of assigned PDWs per RF second.

This widget is intentionally low-rate and display-only. The processing process
counts PDWs assigned to each persistent emitter; the GUI receives only completed
per-second counts in its normal state snapshot.
"""

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PDWRateHistogramCanvas(FigureCanvas):
    HISTORY_SECONDS = 12

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(3.8, 4.2), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumWidth(300)
        self._configure_axes()
        self.clear_plot()

    def _configure_axes(self):
        ax = self.axes
        ax.set_title("PDWs / SECOND", fontsize=10, fontweight="bold")
        ax.set_xlabel("RF time (s)", fontsize=8)
        ax.set_ylabel("PDWs", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)

    def clear_plot(self, message="NO EMITTER SELECTED"):
        self.axes.clear()
        self._configure_axes()
        self.axes.set_xlim(-0.5, self.HISTORY_SECONDS - 0.5)
        self.axes.set_ylim(0, 1)
        self.axes.text(
            0.5, 0.5, message,
            transform=self.axes.transAxes,
            ha="center", va="center", fontsize=9,
        )
        self.draw_idle()

    def update_history(self, emitter_id, history):
        self.axes.clear()
        self._configure_axes()

        items = list(history or [])[-self.HISTORY_SECONDS:]
        if not items:
            self.clear_plot(f"{emitter_id}: WAITING FOR PDWs")
            return

        seconds = [int(item[0]) for item in items]
        counts = np.asarray([int(item[1]) for item in items], dtype=float)

        # Left-pad the first few seconds so the newest observation always enters
        # at the far right and the whole plot moves from right to left like the
        # mode ticker parade.
        pad = self.HISTORY_SECONDS - len(counts)
        plotted_counts = np.concatenate((np.zeros(pad, dtype=float), counts))
        x = np.arange(self.HISTORY_SECONDS)

        self.axes.bar(x, plotted_counts, width=0.78)
        ymax = max(1.0, float(np.max(plotted_counts)) * 1.15)
        self.axes.set_xlim(-0.5, self.HISTORY_SECONDS - 0.5)
        self.axes.set_ylim(0.0, ymax)

        tick_positions = []
        tick_labels = []
        for i, sec in enumerate(seconds):
            pos = pad + i
            if i == len(seconds) - 1 or sec % 2 == 0:
                tick_positions.append(pos)
                tick_labels.append(str(sec))
        self.axes.set_xticks(tick_positions)
        self.axes.set_xticklabels(tick_labels)
        self.axes.set_title(f"{emitter_id}  PDWs / SECOND", fontsize=10, fontweight="bold")

        latest = int(counts[-1])
        self.axes.text(
            0.98, 0.96, f"NOW {latest}/s",
            transform=self.axes.transAxes,
            ha="right", va="top", fontsize=8, fontweight="bold",
        )
        self.draw_idle()
