"""Selected-emitter frequency-versus-time PDW waterfall."""

import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class EmitterFrequencyWaterfallWindow(QMainWindow):
    TIME_SPAN_S = 20.0
    TIME_BINS = 200
    FREQ_BINS = 160

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("S2B ESM - Selected Emitter Frequency Waterfall")
        self.resize(900, 560)

        body = QWidget(self)
        layout = QVBoxLayout(body)

        self.heading = QLabel("NO EMITTER SELECTED")
        layout.addWidget(self.heading)

        # Use fixed axes for the waterfall and colourbar. Repeatedly creating and
        # removing a Matplotlib colourbar changes the main axes geometry, which
        # made the plot shrink on every refresh.
        self.figure = Figure(figsize=(8.5, 4.8))
        grid = self.figure.add_gridspec(
            1, 2,
            width_ratios=(30, 1),
            left=0.10, right=0.94, bottom=0.12, top=0.90,
            wspace=0.12,
        )
        self.axes = self.figure.add_subplot(grid[0, 0])
        self.colorbar_axes = self.figure.add_subplot(grid[0, 1])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.setCentralWidget(body)

        self._image = None
        self._colorbar = None
        self.clear_plot()

    def clear_plot(self, message="NO PDW HISTORY"):
        self.axes.clear()
        self.colorbar_axes.clear()
        self.axes.set_title("FREQUENCY WATERFALL")
        self.axes.set_xlabel("RF time (s)")
        self.axes.set_ylabel("Frequency (MHz)")
        self.axes.text(
            0.5, 0.5, message,
            transform=self.axes.transAxes,
            ha="center", va="center",
        )
        self.colorbar_axes.set_axis_off()
        self.canvas.draw_idle()

    def update_pdws(self, emitter_id, pdw_rows):
        rows = list(pdw_rows or [])
        self.heading.setText(f"{emitter_id}  FREQUENCY WATERFALL")
        if not rows:
            self.clear_plot(f"{emitter_id}: WAITING FOR PDWs")
            return

        data = np.asarray(rows, dtype=float)
        t = data[:, 0]
        f = data[:, 1] / 1e6
        amp = data[:, 2]

        newest = float(np.max(t))
        oldest = max(float(np.min(t)), newest - self.TIME_SPAN_S)
        keep = t >= oldest
        t = t[keep]
        f = f[keep]
        amp = amp[keep]

        fmin = float(np.min(f))
        fmax = float(np.max(f))
        if fmax - fmin < 1.0:
            mid = 0.5 * (fmin + fmax)
            fmin, fmax = mid - 0.5, mid + 0.5
        else:
            pad = max(0.5, 0.08 * (fmax - fmin))
            fmin -= pad
            fmax += pad

        te = np.linspace(oldest, newest + 1e-9, self.TIME_BINS + 1)
        fe = np.linspace(fmin, fmax, self.FREQ_BINS + 1)
        image = np.full((self.FREQ_BINS, self.TIME_BINS), np.nan)

        ti = np.clip(
            np.searchsorted(te, t, side="right") - 1,
            0, self.TIME_BINS - 1,
        )
        fi = np.clip(
            np.searchsorted(fe, f, side="right") - 1,
            0, self.FREQ_BINS - 1,
        )
        for x, y, a in zip(ti, fi, amp):
            if np.isnan(image[y, x]) or a > image[y, x]:
                image[y, x] = a

        vmin = float(np.percentile(amp, 10))
        vmax = float(np.max(amp))
        if vmax <= vmin:
            vmax = vmin + 1.0

        self.axes.clear()
        self.axes.set_title(f"{emitter_id}  FREQUENCY vs TIME")
        self.axes.set_xlabel("RF time (s)")
        self.axes.set_ylabel("Frequency (MHz)")
        self._image = self.axes.imshow(
            image,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[oldest, newest, fmin, fmax],
            vmin=vmin,
            vmax=vmax,
        )

        self.colorbar_axes.clear()
        self.colorbar_axes.set_axis_on()
        self._colorbar = self.figure.colorbar(
            self._image,
            cax=self.colorbar_axes,
        )
        self._colorbar.set_label("Measured strength (dBFS)")
        self.canvas.draw_idle()
