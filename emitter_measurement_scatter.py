"""Pop-out diagnostic plot for the selected streaming emitter.

The scatter deliberately shows pulse-level PRI only:
    x = short inter-pulse interval derived from consecutive measured PDW TOAs
    y = measured pulse amplitude in dBFS
    colour = RF time of the later pulse

Long inter-pulse gaps are not PRI.  They delimit illumination visits and are used
separately to estimate visit-to-visit/revisit timing for the diagnostic header.

The window is diagnostic only; it does not feed association, library matching or
behaviour inference.
"""

import numpy as np

from PyQt5.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize


class EmitterMeasurementScatterWindow(QMainWindow):
    # Anything above 10 ms is treated as a break between pulse trains rather than
    # a pulse PRI for this diagnostic.  This is intentionally a display-side gate;
    # it does not alter the tracker or behaviour engine.
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
        self._configure_axes("SIGNAL STRENGTH vs PRI")
        self.clear_plot()

    def _configure_axes(self, title):
        self.axes.set_title(title)
        self.axes.set_xlabel("Measured PRI (us)")
        self.axes.set_ylabel("Measured signal strength (dBFS)")
        self.axes.grid(True, alpha=0.25)

    def clear_plot(self, message="No selected emitter measurements yet"):
        self._scatter.set_offsets(np.empty((0, 2)))
        self._scatter.set_array(np.asarray([], dtype=float))
        self._norm.vmin = 0.0
        self._norm.vmax = 1.0
        self._scatter.changed()
        self._colorbar.update_normal(self._scatter)
        self._configure_axes("SIGNAL STRENGTH vs PRI")
        self.axes.relim()
        self.axes.autoscale_view()
        if self._message is not None:
            self._message.remove()
        self._message = self.axes.text(0.5, 0.5, message, transform=self.axes.transAxes,
                                       ha="center", va="center")
        self.canvas.draw_idle()

    @classmethod
    def _revisit_period_s(cls, toas):
        """Estimate illumination visit start-to-start period from measured TOAs."""
        if len(toas) < 3:
            return None
        gaps = np.diff(toas)
        break_indices = np.flatnonzero(gaps > cls.MAX_PULSE_PRI_S)
        if len(break_indices) == 0:
            return None
        # First observed pulse starts visit 1; each pulse immediately after a long
        # gap starts the next visit.  Start-to-start differences represent revisit
        # period, unlike the long gap itself which excludes the illumination width.
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
        if len(pdws) < 2:
            self.summary.setText(f"{track.emitter_id}: waiting for enough measured pulses")
            self.clear_plot("Waiting for two or more measured pulses")
            return

        toas = np.asarray([p.toa_s for p in pdws], dtype=float)
        amps = np.asarray([p.amplitude_dbfs for p in pdws], dtype=float)
        intervals_s = np.diff(toas)
        point_amp_all = amps[1:]
        point_time_all = toas[1:]

        # Separate signal-level PRI from behaviour-level illumination gaps.
        pulse_mask = (
            np.isfinite(intervals_s)
            & np.isfinite(point_amp_all)
            & np.isfinite(point_time_all)
            & (intervals_s > 0.0)
            & (intervals_s <= self.MAX_PULSE_PRI_S)
        )
        pri_us = intervals_s[pulse_mask] * 1e6
        point_amp = point_amp_all[pulse_mask]
        point_time = point_time_all[pulse_mask]
        revisit_s = self._revisit_period_s(toas)

        if len(pri_us) == 0:
            self.clear_plot("No pulse-level PRI observations below 10 ms")
            return

        if len(pri_us) > self.max_points:
            pri_us = pri_us[-self.max_points:]
            point_amp = point_amp[-self.max_points:]
            point_time = point_time[-self.max_points:]

        if self._message is not None:
            self._message.remove()
            self._message = None

        offsets = np.column_stack((pri_us, point_amp))
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

        self.axes.set_title(f"{track.emitter_id}  SIGNAL STRENGTH vs PRI")

        xmin = float(np.min(pri_us)); xmax = float(np.max(pri_us))
        ymin = float(np.min(point_amp)); ymax = float(np.max(point_amp))
        xpad = max(1.0, 0.05 * max(xmax - xmin, 1.0))
        ypad = max(0.5, 0.05 * max(ymax - ymin, 1.0))
        self.axes.set_xlim(xmin - xpad, xmax + xpad)
        self.axes.set_ylim(ymin - ypad, ymax + ypad)

        median_pri = float(np.median(pri_us))
        peak_amp = float(np.max(point_amp))
        label = library_id or "UNASSIGNED"
        revisit_text = f"revisit {revisit_s:.3f} s" if revisit_s is not None else "revisit -"
        self.summary.setText(
            f"{track.emitter_id} / {label}   |   points {len(pri_us)}   |   "
            f"median PRI {median_pri:.1f} us   |   {revisit_text}   |   "
            f"peak {peak_amp:.1f} dBFS   |   latest RF time {point_time[-1]:.2f} s"
        )
        self.canvas.draw_idle()
