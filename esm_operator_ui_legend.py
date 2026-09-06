"""S2B ESM operator UI launcher with assessment-state legend and bearing aids.

This keeps the first operator UI intact while adding the colour key, faint 45-degree
bearing spokes and explicit bearing lines to assessed emitters. It can be folded
back into esm_operator_ui.py once the UI layout settles.
"""

import sys
import math

from PyQt5.QtWidgets import QApplication
from matplotlib.lines import Line2D

import esm_operator_ui


_original_configure_axes = esm_operator_ui.PolarEmitterCanvas._configure_axes


def _configure_axes_with_legend(self):
    _original_configure_axes(self)

    # Reinforce the principal 45-degree bearing axes with faint black spokes.
    # These are bearing references only; radial distance still has no range meaning.
    for bearing_deg in range(0, 360, 45):
        angle = math.radians(bearing_deg)
        self.axes.plot(
            [angle, angle],
            [0.0, 1.0],
            color="black",
            linewidth=0.7,
            alpha=0.16,
            zorder=0,
        )

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


def _update_emitters_with_bearing_lines(self, emitters, selected_index=0):
    self._configure_axes()
    ax = self.axes

    if not emitters:
        ax.text(
            0.5,
            0.5,
            "NO EMITTERS",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        self.draw_idle()
        return

    # Bearing-only ESM display: radius is deliberately artificial and is used
    # only to separate labels when several emitters share a bearing. It must not
    # be interpreted as range.
    bearing_counts = {}
    for emitter in emitters:
        key = round(emitter["aoa_deg"], 1)
        bearing_counts[key] = bearing_counts.get(key, 0) + 1

    bearing_seen = {}
    for index, emitter in enumerate(emitters):
        angle = math.radians(emitter["aoa_deg"] % 360.0)
        key = round(emitter["aoa_deg"], 1)
        occurrence = bearing_seen.get(key, 0)
        bearing_seen[key] = occurrence + 1
        count = bearing_counts[key]
        radius = 0.78 if count == 1 else 0.60 + 0.18 * occurrence

        # Explicit line of bearing from ownship/origin to the emitter symbol.
        # A selected emitter gets a slightly stronger line for readability.
        ax.plot(
            [angle, angle],
            [0.0, radius],
            color="black",
            linewidth=1.5 if index == selected_index else 1.0,
            alpha=0.75 if index == selected_index else 0.55,
            zorder=1,
        )

        marker = "o" if index != selected_index else "D"
        size = 90 if index != selected_index else 125
        color = emitter["display_color"]
        ax.scatter(
            [angle],
            [radius],
            s=size,
            marker=marker,
            c=[color],
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(
            angle,
            min(radius + 0.09, 0.98),
            emitter["emitter_id"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=4,
        )

    self.draw_idle()


esm_operator_ui.PolarEmitterCanvas._configure_axes = _configure_axes_with_legend
esm_operator_ui.PolarEmitterCanvas.update_emitters = _update_emitters_with_bearing_lines


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("S2B ESM")
    window = esm_operator_ui.S2BOperatorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
