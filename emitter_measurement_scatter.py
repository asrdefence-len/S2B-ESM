"""Pop-out diagnostic plot for the selected streaming emitter.

Each point represents one measured inter-pulse interval for a tracked emitter:
    x = PRI derived from consecutive measured PDW TOAs
    y = measured pulse amplitude in dBFS
    colour = RF time of the later pulse

The window is diagnostic only; it does not feed association, library matching or
behaviour inference.
"""

import numpy as np

from PyQt5.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize


class EmitterMeasurementScatterWindow(QMainWindow):
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

        # Fixed axes geometry.  Create the scatter and colorbar ONCE and update
        # their data in place.  Recreating Colorbar objects on every Qt refresh
        # can recursively wrap Matplotlib's axes locator and eventually crash.
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
        pri_us = np.diff(toas) * 1e6
        point_amp = amps[1:]
        point_time = toas[1:]

        valid = np.isfinite(pri_us) & np.isfinite(point_amp) & np.isfinite(point_time) & (pri_us > 0.0)
        pri_us = pri_us[valid]
        point_amp = point_amp[valid]
        point_time = point_time[valid]
        if len(pri_us) == 0:
            self.clear_plot("No valid positive PRI observations")
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
        self.axes.relim()
        self.axes.autoscale_view()

        # PathCollection offsets are not always included by relim(), so explicitly
        # bound the current point cloud with a small margin.
        xmin = float(np.min(pri_us)); xmax = float(np.max(pri_us))
        ymin = float(np.min(point_amp)); ymax = float(np.max(point_amp))
        xpad = max(1.0, 0.05 * max(xmax - xmin, 1.0))
        ypad = max(0.5, 0.05 * max(ymax - ymin, 1.0))
        self.axes.set_xlim(xmin - xpad, xmax + xpad)
        self.axes.set_ylim(ymin - ypad, ymax + ypad)

        median_pri = float(np.median(pri_us))
        peak_amp = float(np.max(point_amp))
        label = library_id or "UNASSIGNED"
        self.summary.setText(
            f"{track.emitter_id} / {label}   |   points {len(pri_us)}   |   "
            f"median PRI {median_pri:.1f} us   |   peak {peak_amp:.1f} dBFS   |   "
            f"latest RF time {point_time[-1]:.2f} s"
        )
        self.canvas.draw_idle()
