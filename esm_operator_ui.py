import sys
import math

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from config import *
from simulated_source import SimulatedSource
from pulse_detector import PulseDetector
from pdw_extractor import PDWExtractor
from gated_fast_probabilistic_mht import GatedFastProbabilisticMultipleHypothesisAssociator
from physical_emitter_correlation import PhysicalEmitterCorrelator
from operator_display import OperatorEmitterSummary, _physical_groups
from scenarios import get_scenario, list_scenarios


REFRESH_MS = 750

ASSESSMENT_COLORS = {
    "UNASSESSED": "#777777",
    "MONITOR": "#2f7fbf",
    "CHANGED": "#d28b00",
    "OF INTEREST": "#e56b00",
    "THREAT": "#b22222",
}


class PolarEmitterCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(6.2, 6.2), tight_layout=True)
        self.axes = self.figure.add_subplot(111, projection="polar")
        super().__init__(self.figure)
        self.setParent(parent)
        self._configure_axes()

    def _configure_axes(self):
        ax = self.axes
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks((0.25, 0.50, 0.75, 1.0))
        ax.set_yticklabels(())
        ax.set_thetagrids(range(0, 360, 45))
        ax.grid(True, alpha=0.28)
        ax.set_title("EMITTER BEARING PICTURE", pad=18, fontsize=12, fontweight="bold")

        # Faint cardinal/inter-cardinal bearing reference spokes.
        for bearing_deg in range(0, 360, 45):
            angle = math.radians(bearing_deg)
            ax.plot(
                [angle, angle],
                [0.0, 1.0],
                color="black",
                linewidth=0.7,
                alpha=0.16,
                zorder=0,
            )

        legend_items = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=ASSESSMENT_COLORS["UNASSESSED"], markeredgecolor="black",
                   markersize=8, label="UNASSESSED"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=ASSESSMENT_COLORS["MONITOR"], markeredgecolor="black",
                   markersize=8, label="MONITOR"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=ASSESSMENT_COLORS["CHANGED"], markeredgecolor="black",
                   markersize=8, label="CHANGED"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=ASSESSMENT_COLORS["OF INTEREST"], markeredgecolor="black",
                   markersize=8, label="OF INTEREST"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=ASSESSMENT_COLORS["THREAT"], markeredgecolor="black",
                   markersize=8, label="THREAT"),
        ]
        ax.legend(
            handles=legend_items,
            title="ASSESSMENT",
            loc="upper right",
            bbox_to_anchor=(-0.18, 1.08),
            framealpha=0.92,
            fontsize=8,
            title_fontsize=8,
            borderaxespad=0.0,
        )

    def update_emitters(self, emitters, selected_index=0):
        self._configure_axes()
        ax = self.axes

        if not emitters:
            ax.text(0.5, 0.5, "NO EMITTERS", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
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

            # Explicit line of bearing. The selected emitter is emphasized slightly.
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
                [angle], [radius], s=size, marker=marker, c=[color],
                edgecolors="black", linewidths=0.8, zorder=3,
            )

            # WATCH is an operator attention state, separate from system assessment.
            # Use a gold outline rather than changing the assessment colour.
            if emitter.get("watched", False):
                ax.scatter(
                    [angle], [radius], s=size + 115, marker="o",
                    facecolors="none", edgecolors="#d4a017", linewidths=2.3, zorder=2,
                )

            label = emitter["emitter_id"] + (" *" if emitter.get("watched", False) else "")
            ax.text(
                angle, min(radius + 0.09, 0.98), label,
                ha="center", va="center", fontsize=10, fontweight="bold", zorder=4,
            )

        self.draw_idle()


class S2BOperatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S2B ESM - Operator Display")
        self.resize(1450, 880)

        self.running = False
        self.emitters = []
        self.selected_emitter_index = 0
        self.extractor = None
        self.watched_emitters = set()

        self.timer = QTimer(self)
        self.timer.setInterval(REFRESH_MS)
        self.timer.timeout.connect(self._refresh)

        self._build_ui()
        self._set_status("STOPPED")
        self._refresh()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        title = QLabel("S2B ESM OPERATOR DISPLAY")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        top.addWidget(title)
        top.addStretch(1)

        top.addWidget(QLabel("Scenario:"))
        self.scenario_combo = QComboBox()
        for scenario in list_scenarios():
            self.scenario_combo.addItem(scenario.name)
        index = self.scenario_combo.findText("close_emitters")
        if index >= 0:
            self.scenario_combo.setCurrentIndex(index)
        self.scenario_combo.currentTextChanged.connect(self._scenario_changed)
        top.addWidget(self.scenario_combo)

        self.status_label = QLabel()
        self.status_label.setMinimumWidth(100)
        self.status_label.setAlignment(Qt.AlignCenter)
        top.addWidget(self.status_label)

        self.start_button = QPushButton("START")
        self.start_button.clicked.connect(self.start_system)
        top.addWidget(self.start_button)

        self.stop_button = QPushButton("STOP")
        self.stop_button.clicked.connect(self.stop_system)
        top.addWidget(self.stop_button)

        self.exit_button = QPushButton("EXIT")
        self.exit_button.clicked.connect(self.close)
        top.addWidget(self.exit_button)
        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.polar = PolarEmitterCanvas(left)
        left_layout.addWidget(self.polar, stretch=1)

        self.emitter_table = QTableWidget(0, 7)
        self.emitter_table.setHorizontalHeaderLabels(
            ["Emitter", "AOA", "RF MHz", "Waveform", "State", "Watch", "Track conf."]
        )
        self.emitter_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.emitter_table.setSelectionMode(QTableWidget.SingleSelection)
        self.emitter_table.cellClicked.connect(self._emitter_selected)
        self.emitter_table.horizontalHeader().setStretchLastSection(True)
        self.emitter_table.setMaximumHeight(230)
        left_layout.addWidget(self.emitter_table)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.emitter_heading = QLabel("NO EMITTER SELECTED")
        self.emitter_heading.setStyleSheet("font-size: 17px; font-weight: 700;")
        right_layout.addWidget(self.emitter_heading)

        self.assessment_label = QLabel("UNASSESSED")
        self.assessment_label.setAlignment(Qt.AlignCenter)
        self.assessment_label.setMinimumHeight(38)
        right_layout.addWidget(self.assessment_label)

        self.watch_button = QPushButton("WATCH SELECTED EMITTER")
        self.watch_button.clicked.connect(self._toggle_watch)
        self.watch_button.setEnabled(False)
        right_layout.addWidget(self.watch_button)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 12px;")
        right_layout.addWidget(self.details, stretch=3)

        notes_title = QLabel("OPERATOR / S2B NOTES")
        notes_title.setStyleSheet("font-weight: 700;")
        right_layout.addWidget(notes_title)
        self.notes = QTextEdit()
        self.notes.setPlaceholderText(
            "Operator notes. Later this panel will also show behavioural hypotheses, "
            "missing evidence and suggested probes."
        )
        right_layout.addWidget(self.notes, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([880, 570])

        footer = QLabel(
            "Colour = system assessment. Gold ring/* = operator WATCH. "
            "Radial position is visual separation only and does NOT represent range."
        )
        footer.setStyleSheet("color: #666;")
        root.addWidget(footer)

    def _set_status(self, state):
        self.status_label.setText(state)
        if state == "RUNNING":
            self.status_label.setStyleSheet("background:#1f7a3a;color:white;font-weight:700;padding:6px;")
        elif state == "ERROR":
            self.status_label.setStyleSheet("background:#9b1c1c;color:white;font-weight:700;padding:6px;")
        else:
            self.status_label.setStyleSheet("background:#555;color:white;font-weight:700;padding:6px;")

    def start_system(self):
        if self.running:
            return
        self.running = True
        self._set_status("RUNNING")
        self.timer.start()
        self._refresh()

    def stop_system(self):
        self.running = False
        self.timer.stop()
        self._set_status("STOPPED")

    def closeEvent(self, event):
        self.timer.stop()
        self.running = False
        event.accept()

    def _scenario_changed(self, _name):
        self.selected_emitter_index = 0
        self.extractor = None
        self.watched_emitters.clear()
        self._refresh()

    def _make_mht(self):
        return GatedFastProbabilisticMultipleHypothesisAssociator(
            frequency_sigma_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,
            pulse_width_sigma_s=ASSOCIATION_PULSE_WIDTH_TOLERANCE_S,
            amplitude_sigma_db=ASSOCIATION_AMPLITUDE_TOLERANCE_DB,
            timing_sigma_s=ASSOCIATION_TIMING_TOLERANCE_S,
            beam_width=MHT_BEAM_WIDTH,
            max_emitters=MHT_MAX_EMITTERS,
            birth_probability=PMHT_BIRTH_PROBABILITY,
            clutter_probability=PMHT_CLUTTER_PROBABILITY,
            modulation_match_probability=PMHT_MODULATION_MATCH_PROBABILITY,
            missed_pulse_probability=PMHT_MISSED_PULSE_PROBABILITY,
            max_pri_multiple=PMHT_MAX_PRI_MULTIPLE,
        )

    def _process_snapshot(self):
        scenario = get_scenario(self.scenario_combo.currentText())
        source = SimulatedSource(
            sample_rate_hz=SAMPLE_RATE_HZ,
            center_frequency_hz=CENTER_FREQUENCY_HZ,
            emitters=scenario.emitters,
            noise_std=scenario.noise_std,
        )
        iq, metadata = source.read()
        detector = PulseDetector(
            threshold=DETECTION_THRESHOLD,
            sample_rate_hz=metadata["sample_rate_hz"],
            min_pulse_width_s=MIN_PULSE_WIDTH_S,
        )
        if self.extractor is None:
            self.extractor = PDWExtractor(
                sample_rate_hz=metadata["sample_rate_hz"],
                center_frequency_hz=metadata["center_frequency_hz"],
            )
        else:
            self.extractor.next_pdw_id = 1

        pulses = detector.detect(iq)
        pdws = [self.extractor.extract(iq, pulse) for pulse in pulses]

        mht = self._make_mht()
        hypotheses = mht.associate(pdws)
        if not hypotheses:
            return scenario, [], len(pulses)
        marginals = mht.association_marginals(hypotheses)
        track_membership = mht.reference_track_membership(hypotheses)
        summaries = OperatorEmitterSummary().build(hypotheses, marginals, track_membership)
        physical_hypotheses = PhysicalEmitterCorrelator(
            frequency_scale_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,
            amplitude_scale_db=ASSOCIATION_AMPLITUDE_TOLERANCE_DB,
        ).correlate(hypotheses[0], [])
        groups = _physical_groups(summaries, physical_hypotheses)

        emitters = []
        for emitter_index, group in enumerate(groups, start=1):
            tracks = group["tracks"]
            current = max(tracks, key=lambda item: item["end_toa_s"])
            changed = len(tracks) > 1
            mean_aoa = sum(t.get("aoa_deg", 0.0) for t in tracks) / len(tracks)
            track_conf = sum(t["track_confidence"] for t in tracks) / len(tracks)
            pulse_count = sum(t["pulse_count"] for t in tracks)

            # System assessment is deliberately separate from operator WATCH.
            # OF INTEREST and THREAT are reserved for later behaviour/threat logic.
            if pulse_count < 3 or track_conf < 0.75:
                state = "UNASSESSED"
            elif changed:
                state = "CHANGED"
            else:
                state = "MONITOR"

            emitter_id = f"E{emitter_index}"
            emitters.append({
                "emitter_id": emitter_id,
                "aoa_deg": mean_aoa,
                "state": state,
                "display_color": ASSESSMENT_COLORS[state],
                "watched": emitter_id in self.watched_emitters,
                "tracks": tracks,
                "current": current,
                "links": group["links"],
                "track_confidence": track_conf,
            })

        return scenario, emitters, len(pulses)

    def _refresh(self):
        try:
            scenario, emitters, pulse_count = self._process_snapshot()
            self.emitters = emitters
            if self.selected_emitter_index >= len(emitters):
                self.selected_emitter_index = max(0, len(emitters) - 1)
            self._populate_table()
            self.polar.update_emitters(self.emitters, self.selected_emitter_index)
            self._show_selected_emitter()
            self.statusBar().showMessage(
                f"Scenario: {scenario.name} | Pulses: {pulse_count} | Physical emitters: {len(emitters)}"
            )
        except Exception as exc:
            self.timer.stop()
            self.running = False
            self._set_status("ERROR")
            self.statusBar().showMessage(str(exc))
            self.details.setPlainText(f"UI processing error:\n\n{exc}")

    def _populate_table(self):
        self.emitter_table.setRowCount(len(self.emitters))
        for row, emitter in enumerate(self.emitters):
            current = emitter["current"]
            values = (
                emitter["emitter_id"],
                f"{emitter['aoa_deg']:.1f} deg",
                f"{current['frequency_hz'] / 1e6:.3f}",
                current["modulation"],
                emitter["state"],
                "WATCH" if emitter.get("watched", False) else "",
                f"{100.0 * emitter['track_confidence']:.1f}%",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.emitter_table.setItem(row, column, item)
        if self.emitters:
            self.emitter_table.selectRow(self.selected_emitter_index)

    def _emitter_selected(self, row, _column):
        self.selected_emitter_index = row
        self.polar.update_emitters(self.emitters, self.selected_emitter_index)
        self._show_selected_emitter()

    def _toggle_watch(self):
        if not self.emitters:
            return
        emitter_id = self.emitters[self.selected_emitter_index]["emitter_id"]
        if emitter_id in self.watched_emitters:
            self.watched_emitters.remove(emitter_id)
        else:
            self.watched_emitters.add(emitter_id)
        for emitter in self.emitters:
            emitter["watched"] = emitter["emitter_id"] in self.watched_emitters
        self._populate_table()
        self.polar.update_emitters(self.emitters, self.selected_emitter_index)
        self._show_selected_emitter()

    def _show_selected_emitter(self):
        if not self.emitters:
            self.emitter_heading.setText("NO EMITTER SELECTED")
            self.assessment_label.setText("UNASSESSED")
            self.assessment_label.setStyleSheet("background:#777;color:white;font-weight:700;padding:7px;")
            self.watch_button.setEnabled(False)
            self.details.setPlainText("No physical emitters currently assessed.")
            return

        emitter = self.emitters[self.selected_emitter_index]
        current = emitter["current"]
        watched = emitter.get("watched", False)
        watch_text = "  * WATCH" if watched else ""
        self.emitter_heading.setText(
            f"{emitter['emitter_id']}  |  BEARING {emitter['aoa_deg']:.1f} deg{watch_text}"
        )
        self.assessment_label.setText(emitter["state"])
        self.assessment_label.setStyleSheet(
            f"background:{emitter['display_color']};color:white;font-weight:700;padding:7px;"
        )
        self.watch_button.setEnabled(True)
        self.watch_button.setText("REMOVE WATCH" if watched else "WATCH SELECTED EMITTER")

        pri = "UNRESOLVED" if current["pri_s"] is None else f"{current['pri_s'] * 1e6:.1f} us"
        track_ids = ", ".join(f"T{track['track_id']}" for track in emitter["tracks"])
        lines = [
            "CURRENT OBSERVED STATE",
            "----------------------",
            f"Physical emitter : {emitter['emitter_id']}",
            f"Bearing          : {emitter['aoa_deg']:.1f} deg",
            f"Operator watch   : {'YES' if watched else 'NO'}",
            f"System assessment: {emitter['state']}",
            f"Sequence tracks  : {track_ids}",
            f"RF               : {current['frequency_hz'] / 1e6:.3f} MHz",
            f"PRI median       : {pri}",
            f"PRI pattern      : {current['pri_pattern']}",
            f"Pulse width      : {current['pulse_width_s'] * 1e6:.3f} us",
            f"Waveform family  : {current['modulation']}",
            f"Level            : {current['amplitude_dbfs']:.2f} dBFS",
            f"Pulses           : {current['pulse_count']}",
            f"Track confidence : {100.0 * emitter['track_confidence']:.1f}%",
            "",
            "BEHAVIOUR / CHANGE",
            "------------------",
        ]

        if len(emitter["tracks"]) > 1:
            lines.append("Observable state change detected between linked sequence tracks.")
            for track in emitter["tracks"]:
                tpri = "UNRESOLVED" if track["pri_s"] is None else f"{track['pri_s'] * 1e6:.1f} us"
                lines.append(
                    f"T{track['track_id']}: RF={track['frequency_hz']/1e6:.3f} MHz, "
                    f"PRI={tpri}, PW={track['pulse_width_s']*1e6:.3f} us, MOD={track['modulation']}"
                )
        else:
            lines.append("No significant linked-track behaviour change currently detected.")

        lines.extend([
            "",
            "S2B INTERPRETATION",
            "------------------",
            "Behaviour hypotheses are not enabled yet.",
            "OF INTEREST and THREAT are therefore not assigned automatically yet.",
            "WATCH is an operator attention flag and does not change system assessment.",
        ])
        self.details.setPlainText("\n".join(lines))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("S2B ESM")
    window = S2BOperatorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
