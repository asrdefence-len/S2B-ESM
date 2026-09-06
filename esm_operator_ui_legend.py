"""S2B ESM operator UI launcher with assessment-state legend.

This keeps the first operator UI intact while adding the colour key to the
bearing picture. It can be folded back into esm_operator_ui.py once the UI
layout settles.
"""

import sys

from PyQt5.QtWidgets import QApplication
from matplotlib.lines import Line2D

import esm_operator_ui


_original_configure_axes = esm_operator_ui.PolarEmitterCanvas._configure_axes


def _configure_axes_with_legend(self):
    _original_configure_axes(self)

    legend_items = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#777777",
               markeredgecolor="black", markersize=8, label="UNASSESSED"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#2f7fbf",
               markeredgecolor="black", markersize=8, label="MONITOR"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#d28b00",
               markeredgecolor="black", markersize=8, label="CHANGED"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#e56b00",
               markeredgecolor="black", markersize=8, label="OF INTEREST"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#b22222",
               markeredgecolor="black", markersize=8, label="THREAT"),
    ]

    # Place the assessment key in the unused upper-left margin of the canvas,
    # completely clear of the circular bearing plot.
    self.axes.legend(
        handles=legend_items,
        title="ASSESSMENT",
        loc="upper right",
        bbox_to_anchor=(-0.18, 1.08),
        framealpha=0.92,
        fontsize=8,
        title_fontsize=8,
        borderaxespad=0.0,
    )


esm_operator_ui.PolarEmitterCanvas._configure_axes = _configure_axes_with_legend


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("S2B ESM")
    window = esm_operator_ui.S2BOperatorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
