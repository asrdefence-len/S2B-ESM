import sys
import math

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from emitter_library import EmitterLibrary
from mode_history import ObservedModeHistory
from esm_operator_ui import S2BOperatorWindow, PolarEmitterCanvas


class EnhancedPolarEmitterCanvas(PolarEmitterCanvas):
    def update_emitters(self, emitters, selected_index=0):
        super().update_emitters(emitters, selected_index)
        ax = self.axes
        # Put the compact library identity beside, rather than on top of, the symbol.
        counts = {}
        seen = {}
        for e in emitters:
            k = round(e["aoa_deg"], 1)
            counts[k] = counts.get(k, 0) + 1
        for e in emitters:
            a = math.radians(e["aoa_deg"] % 360)
            k = round(e["aoa_deg"], 1)
            n = seen.get(k, 0)
            seen[k] = n + 1
            r = .78 if counts[k] == 1 else .60 + .18 * n
            lib = e.get("library_id", "UNKNOWN")

            # Offset primarily in angle so the text sits alongside the symbol.
            # Choose the side by bearing so labels tend to remain inside the plot.
            bearing = e["aoa_deg"] % 360
            offset_deg = -5.0 if 0 <= bearing < 180 else 5.0
            label_a = a + math.radians(offset_deg)
            ha = "right" if offset_deg < 0 else "left"
            ax.text(
                label_a, r, lib,
                ha=ha, va="center", fontsize=5.5, alpha=.78,
                fontweight="normal", zorder=4,
            )
        self.draw_idle()


class EnhancedS2BOperatorWindow(S2BOperatorWindow):
    def __init__(self):
        self.emitter_library = EmitterLibrary()
        self.mode_history = ObservedModeHistory(max_entries=18)
        super().__init__()
        self.setWindowTitle("S2B ESM - Operator Display")
        self.resize(1750, 1050)
        self.setMinimumSize(1250, 760)

    def _build_ui(self):
        # Build the proven base UI first, then replace/enhance only the left-side displays.
        super()._build_ui()
        old_polar = self.polar
        parent = old_polar.parentWidget()
        layout = parent.layout()
        index = layout.indexOf(old_polar)
        layout.removeWidget(old_polar)
        old_polar.setParent(None)
        self.polar = EnhancedPolarEmitterCanvas(parent)
        layout.insertWidget(index, self.polar, stretch=1)

        # Add Library column to the existing emitter table.
        self.emitter_table.setColumnCount(8)
        self.emitter_table.setHorizontalHeaderLabels([
            "Emitter", "Library", "AOA", "RF MHz", "Waveform", "State", "Watch", "Track conf."
        ])

        # Add a compact, one-second observed-mode waterfall immediately above the table.
        self.mode_history_title = QLabel("SELECTED EMITTER - OBSERVED MODE HISTORY (1 s cells, oldest -> newest)")
        self.mode_history_title.setStyleSheet("font-weight:700; margin-top:4px;")
        self.mode_history_table = QTableWidget(1, 18)
        self.mode_history_table.setVerticalHeaderLabels(["MODE"])
        self.mode_history_table.horizontalHeader().setVisible(False)
        self.mode_history_table.setFixedHeight(72)
        self.mode_history_table.setSelectionMode(QTableWidget.NoSelection)
        self.mode_history_table.setFocusPolicy(Qt.NoFocus)
        table_index = layout.indexOf(self.emitter_table)
        layout.insertWidget(table_index, self.mode_history_title)
        layout.insertWidget(table_index + 1, self.mode_history_table)

    def reset_system(self):
        self.mode_history.clear()
        super().reset_system()

    def _scenario_changed(self, name):
        self.mode_history.clear()
        super()._scenario_changed(name)

    def _assign_library(self, emitter):
        c = emitter["current"]
        illumination = emitter.get("illumination")
        illumination_state = illumination.state if illumination is not None else None
        match = self.emitter_library.identify(
            c.get("frequency_hz"), c.get("pri_s"), c.get("modulation"), illumination_state
        )
        emitter["library_id"] = match.emitter_type
        emitter["library_confidence"] = match.confidence
        emitter["library_reason"] = match.reason
        return emitter

    def _refresh(self):
        super()._refresh()
        if not self.emitters:
            return
        for e in self.emitters:
            self._assign_library(e)
        e3 = next((e for e in self.emitters if e["emitter_id"] == "E3"), None)
        if e3 is not None and e3.get("illumination") is not None:
            t = max(0.0, self.nav_last_sample_s - 0.010)
            self.mode_history.update("E3", t, e3["illumination"].state)
        self._populate_table()
        self.polar.update_emitters(self.emitters, self.selected_emitter_index)
        self._show_selected_emitter()
        self._update_mode_history_display()

    def _populate_table(self):
        self.emitter_table.setRowCount(len(self.emitters))
        for row, e in enumerate(self.emitters):
            c = e["current"]
            vals = (
                e["emitter_id"], e.get("library_id", "UNKNOWN"), f"{e['aoa_deg']:.1f} deg",
                f"{c['frequency_hz']/1e6:.3f}", c["modulation"], e["state"],
                "WATCH" if e.get("watched", False) else "", f"{100*e['track_confidence']:.1f}%",
            )
            for col, value in enumerate(vals):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.emitter_table.setItem(row, col, item)
        if self.emitters:
            self.emitter_table.selectRow(self.selected_emitter_index)

    def _emitter_selected(self, row, column):
        super()._emitter_selected(row, column)
        self._update_mode_history_display()

    def _update_mode_history_display(self):
        if not hasattr(self, "mode_history_table"):
            return
        self.mode_history_table.clearContents()
        if not self.emitters:
            return
        e = self.emitters[self.selected_emitter_index]
        labels = self.mode_history.labels(e["emitter_id"])
        padded = [""] * (18 - len(labels)) + labels[-18:]
        for col, label in enumerate(padded):
            item = QTableWidgetItem(label)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if label == "SEARCH":
                item.setBackground(Qt.lightGray)
            elif label == "DWELL":
                item.setBackground(Qt.yellow)
            self.mode_history_table.setItem(0, col, item)
        self.mode_history_title.setText(
            f"{e['emitter_id']} / {e.get('library_id','UNKNOWN')} - OBSERVED MODE HISTORY (1 s cells, oldest -> newest)"
        )

    def _show_selected_emitter(self):
        super()._show_selected_emitter()
        if not self.emitters:
            return
        e = self.emitters[self.selected_emitter_index]
        lib = e.get("library_id", "UNKNOWN")
        conf = 100.0 * e.get("library_confidence", 0.0)
        reason = e.get("library_reason", "No library evidence")
        existing = self.details.toPlainText()
        prefix = (
            "EMITTER LIBRARY\n"
            "---------------\n"
            f"Library ID       : {lib}\n"
            f"Library confidence: {conf:.1f}%\n"
            f"Match evidence   : {reason}\n\n"
        )
        self.details.setPlainText(prefix + existing)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("S2B ESM")
    window = EnhancedS2BOperatorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
