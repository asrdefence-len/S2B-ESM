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

        # Use an explicit fixed axes rectangle rather than tight_layout.  Repeated
        # removal/recreation of a colorbar under tight_layout progressively stole
        # width from the main axes on every live refresh.
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.axes = self.figure.add_axes([0.10, 0.12, 0.72, 0.80])
        self._colorbar_axes = self.figure.add_axes([0.86, 0.12, 0.025, 0.80])
        self._colorbar_axes.set_visible(False)
        self._colorbar = None
        self.clear_plot()

    def clear_plot(self, message="No selected emitter measurements yet"):
        self.axes.clear()
        self.axes.set_title("SIGNAL STRENGTH vs PRI")
        self.axes.set_xlabel("Measured PRI (us)")
        self.axes.set_ylabel("Measured signal strength (dBFS)")
        self.axes.grid(True, alpha=0.25)
        self.axes.text(0.5, 0.5, message, transform=self.axes.transAxes,
                       ha="center", va="center")
        self._clear_colorbar()
        self.canvas.draw_idle()

    def _clear_colorbar(self):
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None
        self._colorbar_axes.clear()
        self._colorbar_axes.set_visible(False)

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

        valid = np.isfinite(pri_us) & np.isfinite(point_amp) & (pri_us > 0.0)
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

        self.axes.clear()
        scatter = self.axes.scatter(pri_us, point_amp, c=point_time, s=15, alpha=0.75)
        self.axes.set_title(f"{track.emitter_id}  SIGNAL STRENGTH vs PRI")
        self.axes.set_xlabel("Measured PRI (us)")
        self.axes.set_ylabel("Measured signal strength (dBFS)")
        self.axes.grid(True, alpha=0.25)

        # Reuse a dedicated colorbar axes.  The main scatter axes therefore keeps
        # exactly the same geometry regardless of how many live updates occur.
        self._clear_colorbar()
        self._colorbar_axes.set_visible(True)
        self._colorbar = self.figure.colorbar(scatter, cax=self._colorbar_axes)
        self._colorbar.set_label("RF time (s)")

        median_pri = float(np.median(pri_us))
        peak_amp = float(np.max(point_amp))
        label = library_id or "UNASSIGNED"
        self.summary.setText(
            f"{track.emitter_id} / {label}   |   points {len(pri_us)}   |   "
            f"median PRI {median_pri:.1f} us   |   peak {peak_amp:.1f} dBFS   |   "
            f"latest RF time {point_time[-1]:.2f} s"
        )
        self.canvas.draw_idle()
