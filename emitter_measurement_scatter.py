"""Pop-out diagnostic plot for the selected streaming emitter.

The scatter shows track-level PRI state rather than arbitrary adjacent-PDW DTOA:
    x = inferred fundamental PRI for the established emitter track
    y = measured pulse width
    colour = RF time

The track estimator explains missed-pulse gaps as integer multiples of a PRI and
requires persistence before accepting a different PRI as a new state. Long gaps
remain illumination/revisit evidence and are not plotted as PRI.
"""

import numpy as np

from PyQt5.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize


class EmitterMeasurementScatterWindow(QMainWindow):
    MAX_PULSE_PRI_S = 0.010

    def __init__(self, parent=None, max_points=2000):
        super().__init__(parent)
        self.max_points = int(max_points)
        self.setWindowTitle("S2B Emitter Measurement Plot")
        self.resize(850, 650)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.summary = QLabel("Select an emitter in the main display.")
        self.summary.setStyleSheet("font-weight: 600; padding: 4px;")
        layout.addWidget(self.summary)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        self.axes = self.figure.add_axes([0.10, 0.12, 0.72, 0.80])
        self._colorbar_axes = self.figure.add_axes([0.86, 0.12, 0.025, 0.80])
        self._norm = Normalize(vmin=0.0, vmax=1.0)
        self._scatter = self.axes.scatter([], [], c=[], s=15, alpha=0.75, norm=self._norm)
        self._colorbar = self.figure.colorbar(self._scatter, cax=self._colorbar_axes)
        self._colorbar.set_label("RF time (s)")
        self._message = None
        self._configure_axes("PRI STATE vs PULSE WIDTH")
        self.clear_plot()

    def _configure_axes(self, title):
        self.axes.set_title(title)
        self.axes.set_xlabel("Inferred track PRI (us)")
        self.axes.set_ylabel("Measured pulse width (us)")
        self.axes.grid(True, alpha=0.25)

    def clear_plot(self, message="No selected emitter PRI states yet"):
        self._scatter.set_offsets(np.empty((0, 2)))
        self._scatter.set_array(np.asarray([], dtype=float))
        self._norm.vmin = 0.0
        self._norm.vmax = 1.0
        self._scatter.changed()
        self._colorbar.update_normal(self._scatter)
        self._configure_axes("PRI STATE vs PULSE WIDTH")
        self.axes.relim()
        self.axes.autoscale_view()
        if self._message is not None:
            self._message.remove()
        self._message = self.axes.text(0.5, 0.5, message, transform=self.axes.transAxes,
                                       ha="center", va="center")
        self.canvas.draw_idle()

    @classmethod
    def _revisit_period_s(cls, toas):
        if len(toas) < 3:
            return None
        gaps = np.diff(toas)
        break_indices = np.flatnonzero(gaps > cls.MAX_PULSE_PRI_S)
        if len(break_indices) == 0:
            return None
        starts = np.concatenate(([toas[0]], toas[break_indices + 1]))
        if len(starts) < 2:
            return None
        periods = np.diff(starts)
        periods = periods[np.isfinite(periods) & (periods > cls.MAX_PULSE_PRI_S)]
        return float(np.median(periods)) if len(periods) else None

    def update_track(self, track, library_id=None):
        if track is None:
            self.summary.setText("Select an emitter in the main display.")
            self.clear_plot()
            return

        pdws = list(track.pdws)
        history = list(getattr(track, "pri_state_history", []))
        if not history:
            self.summary.setText(f"{track.emitter_id}: waiting for a resolved PRI state")
            self.clear_plot("Waiting for enough pulses to resolve track PRI")
            return

        history = history[-self.max_points:]
        point_time = np.asarray([h[0] for h in history], dtype=float)
        pri_us = np.asarray([h[1] * 1e6 for h in history], dtype=float)
        pw_us = np.asarray([h[2] * 1e6 for h in history], dtype=float)
        confidence = np.asarray([h[3] for h in history], dtype=float)
        mask = np.isfinite(point_time) & np.isfinite(pri_us) & np.isfinite(pw_us) & (pri_us > 0.0)
        point_time = point_time[mask]
        pri_us = pri_us[mask]
        pw_us = pw_us[mask]
        confidence = confidence[mask]

        if len(pri_us) == 0:
            self.clear_plot("No resolved track PRI states")
            return

        if self._message is not None:
            self._message.remove()
            self._message = None

        offsets = np.column_stack((pri_us, pw_us))
        self._scatter.set_offsets(offsets)
        self._scatter.set_array(point_time)

        tmin = float(np.min(point_time))
        tmax = float(np.max(point_time))
        if tmax <= tmin:
            tmax = tmin + 1e-9
        self._norm.vmin = tmin
        self._norm.vmax = tmax
        self._scatter.changed()
        self._colorbar.update_normal(self._scatter)

        self.axes.set_title(f"{track.emitter_id}  PRI STATE vs PULSE WIDTH")
        xmin = float(np.min(pri_us)); xmax = float(np.max(pri_us))
        ymin = float(np.min(pw_us)); ymax = float(np.max(pw_us))
        xpad = max(1.0, 0.05 * max(xmax - xmin, 1.0))
        ypad = max(0.05, 0.05 * max(ymax - ymin, 0.1))
        self.axes.set_xlim(xmin - xpad, xmax + xpad)
        self.axes.set_ylim(ymin - ypad, ymax + ypad)

        toas = np.asarray([p.toa_s for p in pdws], dtype=float) if pdws else np.asarray([])
        revisit_s = self._revisit_period_s(toas)
        label = library_id or "UNASSIGNED"
        revisit_text = f"revisit {revisit_s:.3f} s" if revisit_s is not None else "revisit -"
        current_pri = float(pri_us[-1])
        current_pw = float(pw_us[-1])
        current_conf = float(confidence[-1]) if len(confidence) else 0.0
        self.summary.setText(
            f"{track.emitter_id} / {label}   |   states {len(pri_us)}   |   "
            f"PRI {current_pri:.1f} us ({100.0 * current_conf:.0f}%)   |   "
            f"PW {current_pw:.2f} us   |   {revisit_text}   |   "
            f"latest RF time {point_time[-1]:.2f} s"
        )
        self.canvas.draw_idle()
