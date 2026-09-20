"""Selected-emitter spectrum plus a continuously scrolling waterfall."""

import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class EmitterFrequencyWaterfallWindow(QMainWindow):
    TIME_SPAN_S = 20.0
    TIME_BINS = 160
    FREQ_BINS = 180
    SPECTRUM_LOOKBACK_S = 1.0
    NOISE_FLOOR_DBFS = -55.0
    NOISE_JITTER_DB = 2.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("S2B ESM - Selected Emitter Frequency Waterfall")
        self.resize(900, 650)

        body = QWidget(self)
        layout = QVBoxLayout(body)
        self.heading = QLabel("NO EMITTER SELECTED")
        self.heading.setStyleSheet("font-weight:700;")
        layout.addWidget(self.heading)

        self.figure = Figure(figsize=(8.5, 5.8), facecolor="black")
        grid = self.figure.add_gridspec(
            2, 2,
            height_ratios=(1, 4),
            width_ratios=(30, 1),
            left=.10, right=.94, bottom=.09, top=.93,
            hspace=.08, wspace=.12,
        )
        self.spectrum_axes = self.figure.add_subplot(grid[0, 0])
        self.waterfall_axes = self.figure.add_subplot(grid[1, 0], sharex=self.spectrum_axes)
        self.colorbar_axes = self.figure.add_subplot(grid[1, 1])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.setCentralWidget(body)

        self._rng = np.random.default_rng(24680)
        self._last_emitter_id = None
        self._last_vmin = self.NOISE_FLOOR_DBFS - 5.0
        self._last_vmax = 0.0

        self._style_dark_axes(self.spectrum_axes)
        self._style_dark_axes(self.waterfall_axes)
        self.colorbar_axes.set_facecolor("black")

        # Artists are created ONCE. Recreating a colorbar on every 100 ms update
        # caused Matplotlib's locator chain to recurse until it crashed.
        initial = np.full(
            (self.TIME_BINS, self.FREQ_BINS),
            self.NOISE_FLOOR_DBFS,
            dtype=float,
        )
        self._image = self.waterfall_axes.imshow(
            initial,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            extent=[9403.0, 9413.0, self.TIME_SPAN_S, 0.0],
            vmin=self._last_vmin,
            vmax=self._last_vmax,
            cmap="inferno",
        )
        (self._spectrum_line,) = self.spectrum_axes.plot([], [])

        self._colorbar = self.figure.colorbar(self._image, cax=self.colorbar_axes)
        self._colorbar.set_label("Measured strength (dBFS)", color="white")
        self._colorbar.ax.tick_params(colors="white")
        self._colorbar.outline.set_edgecolor("white")

        self.spectrum_axes.set_title("SELECTED EMITTER SPECTRUM")
        self.spectrum_axes.set_ylabel("Strength (dBFS)")
        self.spectrum_axes.tick_params(labelbottom=False)
        self.waterfall_axes.set_xlabel("Frequency (MHz)")
        self.waterfall_axes.set_ylabel("Time ago (s)")
        self._message = self.waterfall_axes.text(
            .5, .5, "NO PDW HISTORY",
            transform=self.waterfall_axes.transAxes,
            ha="center", va="center", color="white",
        )
        self.canvas.draw_idle()

    @staticmethod
    def _style_dark_axes(ax):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    def clear_plot(self, message="NO PDW HISTORY"):
        self.heading.setText("NO EMITTER SELECTED")
        self._message.set_text(message)
        self._message.set_visible(True)
        self._spectrum_line.set_data([], [])
        self._image.set_data(
            np.full(
                (self.TIME_BINS, self.FREQ_BINS),
                self.NOISE_FLOOR_DBFS,
                dtype=float,
            )
        )
        self.canvas.draw_idle()

    def update_pdws(self, emitter_id, pdw_rows, now_s=None):
        rows = list(pdw_rows or [])
        self.heading.setText(f"{emitter_id}  FREQUENCY WATERFALL")

        if not rows:
            self._message.set_text(f"{emitter_id}: WAITING FOR PDWs")
            self._message.set_visible(True)
            self.canvas.draw_idle()
            return

        data = np.asarray(rows, dtype=float)
        t = data[:, 0]
        f = data[:, 1] / 1e6
        amp = data[:, 2]

        # Scroll against receiver/processing time, not the timestamp of the newest
        # PDW. Therefore silence continues moving downward naturally.
        display_now = float(now_s) if now_s is not None else float(np.max(t))
        oldest = display_now - self.TIME_SPAN_S
        keep = (t >= oldest) & (t <= display_now + 1e-9)
        t = t[keep]
        f = f[keep]
        amp = amp[keep]

        if len(t) == 0:
            self._message.set_text(f"{emitter_id}: NO PDWs IN LAST {self.TIME_SPAN_S:.0f} s")
            self._message.set_visible(True)
            self.canvas.draw_idle()
            return

        # Stable RF display for the E3 experiment; for another emitter, derive a
        # stable local span around its current observed frequencies.
        if np.min(f) >= 9403.0 and np.max(f) <= 9413.0:
            fmin, fmax = 9403.0, 9413.0
        else:
            fmin = float(np.min(f))
            fmax = float(np.max(f))
            if fmax - fmin < 2.0:
                mid = .5 * (fmin + fmax)
                fmin, fmax = mid - 1.0, mid + 1.0
            else:
                pad = max(.5, .08 * (fmax - fmin))
                fmin -= pad
                fmax += pad

        f_edges = np.linspace(fmin, fmax, self.FREQ_BINS + 1)
        centres = .5 * (f_edges[:-1] + f_edges[1:])
        age = display_now - t
        age_edges = np.linspace(0.0, self.TIME_SPAN_S, self.TIME_BINS + 1)

        image = self.NOISE_FLOOR_DBFS + self._rng.normal(
            0.0, .8, size=(self.TIME_BINS, self.FREQ_BINS)
        )
        fi = np.clip(
            np.searchsorted(f_edges, f, side="right") - 1,
            0, self.FREQ_BINS - 1,
        )
        ai = np.clip(
            np.searchsorted(age_edges, age, side="right") - 1,
            0, self.TIME_BINS - 1,
        )
        for y, x, a in zip(ai, fi, amp):
            if a > image[y, x]:
                image[y, x] = a

        vmax = float(np.max(amp))
        vmin = min(self.NOISE_FLOOR_DBFS, float(np.percentile(amp, 10)) - 12.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        self._last_vmin, self._last_vmax = vmin, vmax

        spectrum = self.NOISE_FLOOR_DBFS + self._rng.normal(
            0.0, self.NOISE_JITTER_DB, self.FREQ_BINS
        )
        recent = age <= self.SPECTRUM_LOOKBACK_S
        if np.any(recent):
            for x, a in zip(fi[recent], amp[recent]):
                spectrum[x] = max(spectrum[x], a)
                for offset, drop_db in ((-2, 8.0), (-1, 4.0), (1, 4.0), (2, 8.0)):
                    j = x + offset
                    if 0 <= j < self.FREQ_BINS:
                        spectrum[j] = max(spectrum[j], a - drop_db)

        # Update existing artists only: no clear(), no new imshow(), no new colorbar.
        self._spectrum_line.set_data(centres, spectrum)
        self.spectrum_axes.set_xlim(fmin, fmax)
        self.spectrum_axes.set_ylim(vmin - 3.0, vmax + 3.0)
        self.spectrum_axes.set_title(
            f"{emitter_id}  SPECTRUM (last {self.SPECTRUM_LOOKBACK_S:.0f} s)"
        )

        self._image.set_data(image)
        self._image.set_extent([fmin, fmax, self.TIME_SPAN_S, 0.0])
        self._image.set_clim(vmin=vmin, vmax=vmax)
        self.waterfall_axes.set_xlim(fmin, fmax)
        self.waterfall_axes.set_ylim(self.TIME_SPAN_S, 0.0)

        self._message.set_visible(False)
        self.canvas.draw_idle()
