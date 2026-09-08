"""S2B ESM live operator display with processing isolated from Qt/Matplotlib.

One command still launches the system.  The GUI process now renders low-rate
snapshots only; continuous 40 MS/s processing, emitter tracking and illumination
behaviour run in a dedicated processing process, which owns the existing four
classifier worker processes.
"""

import sys
from types import SimpleNamespace

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QPushButton

from esm_operator_ui_enhanced import EnhancedS2BOperatorWindow
from emitter_measurement_scatter import EmitterMeasurementScatterWindow
from illumination_behaviour import IlluminationAssessment
from operator_processing_engine import OperatorProcessingEngine


class LiveS2BOperatorWindow(EnhancedS2BOperatorWindow):
    WORKERS = 4
    IQ_QUEUE_DEPTH = 32
    CLASSIFIER_QUEUE_DEPTH = 512
    RESULT_QUEUE_DEPTH = 512
    PLOT_REFRESH_MS = 1000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("S2B ESM - Live Parallel 40 MS/s Operator Display")
        self.timer.setInterval(100)
        self.processing_engine = None
        self.snapshot_tracks = {}
        self.latest_processing_status = {}
        self.measurement_plot = None
        self.measurement_plot_timer = QTimer(self)
        self.measurement_plot_timer.setInterval(self.PLOT_REFRESH_MS)
        self.measurement_plot_timer.timeout.connect(self._update_measurement_plot)
        root = self.centralWidget().layout()
        top = root.itemAt(0).layout() if root is not None and root.count() else None
        self.measurement_plot_button = QPushButton("PRI / SIGNAL PLOT")
        self.measurement_plot_button.clicked.connect(self._open_measurement_plot)
        if top is not None:
            top.addWidget(self.measurement_plot_button)

    def _new_stream(self):
        self._shutdown_engine()
        self.processing_engine = OperatorProcessingEngine(
            sample_rate_hz=40_000_000,
            center_frequency_hz=9_415_000_000,
            block_samples=self.BLOCK_SAMPLES,
            noise_std=0.02,
            workers=self.WORKERS,
            iq_queue_depth=self.IQ_QUEUE_DEPTH,
            classifier_queue_depth=self.CLASSIFIER_QUEUE_DEPTH,
            result_queue_depth=self.RESULT_QUEUE_DEPTH,
            frequency_gate_hz=2_000_000.0,
        )
        self.snapshot_tracks = {}
        self.latest_processing_status = {}
        self.last_stream_time_s = 0.0
        # These old in-process members are deliberately unused in live mode.
        self.stream_source = None
        self.stream_processor = None
        self.stream_tracker = None

    def _shutdown_engine(self):
        engine = getattr(self, "processing_engine", None)
        if engine is not None:
            engine.shutdown()
        self.processing_engine = None

    @staticmethod
    def _display_aoa_snapshot(snapshot):
        face = int(snapshot.get("receiver_face", 0) or 0)
        if face:
            return {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}.get(face, 45.0)
        frequency_hz = float(snapshot.get("frequency_hz", 0.0))
        return 135.0 if abs(frequency_hz - 9_410_000_000) <= 2_000_000 else 45.0

    @staticmethod
    def _plot_track(snapshot):
        """Build a lightweight plot view; no full PDW objects cross processes."""
        pdws = [SimpleNamespace(toa_s=t) for t in snapshot.get("recent_toas", [])]
        return SimpleNamespace(
            emitter_id=snapshot["emitter_id"],
            pdws=pdws,
            pri_state_history=snapshot.get("pri_state_history", []),
        )

    def _selected_track(self):
        if not self.emitters:
            return None
        eid = self.emitters[self.selected_emitter_index]["emitter_id"]
        snapshot = self.snapshot_tracks.get(eid)
        return self._plot_track(snapshot) if snapshot is not None else None

    def _open_measurement_plot(self):
        if self.measurement_plot is None:
            self.measurement_plot = EmitterMeasurementScatterWindow(self, max_points=4000)
        self.measurement_plot.show()
        self.measurement_plot.raise_()
        self.measurement_plot.activateWindow()
        self._update_measurement_plot()
        if not self.measurement_plot_timer.isActive():
            self.measurement_plot_timer.start()

    def _update_measurement_plot(self):
        if self.measurement_plot is None:
            return
        if not self.measurement_plot.isVisible():
            self.measurement_plot_timer.stop()
            return
        track = self._selected_track()
        lib = None
        if self.emitters:
            lib = self.emitters[self.selected_emitter_index].get("library_id")
        self.measurement_plot.update_track(track, lib)

    def start_system(self):
        if self.running:
            return
        if self.processing_engine is None:
            self._new_stream()
        self.processing_engine.start()
        self.running = True
        self._set_status("RUNNING")
        self.timer.start()
        self._refresh()
        if (self.measurement_plot is not None and self.measurement_plot.isVisible()
                and not self.measurement_plot_timer.isActive()):
            self.measurement_plot_timer.start()

    def stop_system(self):
        if not self.running:
            return
        self.running = False
        self.timer.stop()
        self.measurement_plot_timer.stop()
        if self.processing_engine is not None:
            self.processing_engine.stop()
        self._set_status("STOPPED")
        status = self.latest_processing_status
        self.statusBar().showMessage(
            f"Stopped | PDWs {status.get('completed_pdws', 0)} | "
            f"IQ high-water {status.get('iq_high_water', 0)}/{status.get('iq_queue_depth', self.IQ_QUEUE_DEPTH)} | "
            f"drops {status.get('drops', 0)}"
        )

    def reset_system(self):
        self.timer.stop()
        self.measurement_plot_timer.stop()
        self.running = False
        self._shutdown_engine()
        self.mode_history.clear()
        self.library_memory.clear()
        self.library_degraded.clear()
        self.library_operator_confirmed.clear()
        self.watched_emitters.clear()
        self.operator_assessments.clear()
        self._details_emitter_id = None
        self._new_stream()
        self._show_prestart_blank()
        self._set_status("STOPPED")
        self._update_measurement_plot()

    def _refresh(self):
        if not self.running or self.processing_engine is None:
            return
        try:
            snapshot = self.processing_engine.latest_snapshot()
            if snapshot is None:
                self.statusBar().showMessage("PROCESSING ENGINE | waiting for first state snapshot")
                return

            status = snapshot.get("status", {})
            self.latest_processing_status = status
            now_s = float(status.get("completed_time_s", self.last_stream_time_s))
            tracks = snapshot.get("tracks", [])
            self.snapshot_tracks = {t["emitter_id"]: t for t in tracks}

            out = []
            for track in tracks:
                current = track["current"]
                illum = IlluminationAssessment(**track["illumination"])
                system_state = (
                    illum.system_assessment if illum.state != "UNRESOLVED"
                    else ("MONITOR" if track["total_pulses"] >= 3 else "UNASSESSED")
                )
                eid = track["emitter_id"]
                e = {
                    "emitter_id": eid,
                    "aoa_deg": self._display_aoa_snapshot(track),
                    "system_state": system_state,
                    "state": system_state,
                    "display_color": self._assessment_color(system_state),
                    "watched": eid in self.watched_emitters,
                    "tracks": [],
                    "current": current,
                    "links": [],
                    "track_confidence": min(1.0, track["total_pulses"] / 20.0),
                    "illumination": illum,
                    "last_seen_s": track["last_seen_s"],
                }
                self._apply_operator_assessment(e)
                self._assign_library(e)
                out.append(e)
                if illum.state in ("PERIODIC_SCAN", "PERSISTENT_ILLUMINATION"):
                    self.mode_history.update(eid, max(0.0, now_s - 0.010), illum.state)

            self.emitters = out
            if self.selected_emitter_index >= len(out):
                self.selected_emitter_index = max(0, len(out) - 1)
            self._populate_table()
            self.polar.update_emitters(out, self.selected_emitter_index)
            self._show_selected_emitter()
            self._update_mode_history_display()
            self.last_stream_time_s = now_s

            self.statusBar().showMessage(
                f"ENGINE + 4 WORKERS | STREAM 40.0 MS/s | RF {status.get('stream_time_s', 0.0):6.2f} s | "
                f"safe {now_s:6.2f} s | lag {1000 * status.get('pipeline_lag_s', 0.0):.0f} ms | "
                f"PDWs {status.get('completed_pdws', 0)} | IQ Q {status.get('iq_queue', 0)}/{status.get('iq_queue_depth', self.IQ_QUEUE_DEPTH)} "
                f"(max {status.get('iq_high_water', 0)}) | CLASS backlog {status.get('classifier_backlog', 0)}/{status.get('classifier_queue_depth', self.CLASSIFIER_QUEUE_DEPTH)} | "
                f"DROPS {status.get('drops', 0)}"
            )
        except Exception as exc:
            self.timer.stop()
            self.measurement_plot_timer.stop()
            self.running = False
            self._shutdown_engine()
            self._set_status("ERROR")
            self.details.setPlainText(f"Live processing/UI boundary error:\n\n{exc}")
            self.statusBar().showMessage(str(exc))

    def _emitter_selected(self, row, column):
        super()._emitter_selected(row, column)
        self._update_measurement_plot()

    @staticmethod
    def _assessment_color(state):
        from esm_operator_ui import ASSESSMENT_COLORS
        return ASSESSMENT_COLORS.get(state, ASSESSMENT_COLORS["UNASSESSED"])

    def closeEvent(self, event):
        self.timer.stop()
        self.measurement_plot_timer.stop()
        self.running = False
        self._shutdown_engine()
        if self.measurement_plot is not None:
            self.measurement_plot.close()
        event.accept()


def main():
    import multiprocessing as mp
    mp.freeze_support()
    app = QApplication(sys.argv)
    app.setApplicationName("S2B ESM")
    window = LiveS2BOperatorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
