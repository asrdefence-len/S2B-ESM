import sys
import math

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QTableWidget, QTableWidgetItem, QPushButton

from emitter_library import EmitterLibrary
from mode_history import ObservedModeHistory
from illumination_behaviour import IlluminationBehaviourManager
from esm_operator_ui import S2BOperatorWindow, PolarEmitterCanvas, ASSESSMENT_COLORS
from simulated_streaming_source import SimulatedStreamingIQSource
from streaming_esm_processor import StreamingESMProcessor
from streaming_emitter_tracker import StreamingEmitterTracker


class EnhancedPolarEmitterCanvas(PolarEmitterCanvas):
    def update_emitters(self, emitters, selected_index=0):
        super().update_emitters(emitters, selected_index)
        ax=self.axes
        counts={}; seen={}
        for e in emitters:
            k=round(e["aoa_deg"],1); counts[k]=counts.get(k,0)+1
        for e in emitters:
            a=math.radians(e["aoa_deg"]%360); k=round(e["aoa_deg"],1); n=seen.get(k,0); seen[k]=n+1
            r=.78 if counts[k]==1 else .60+.18*n
            lib=e.get("library_id","UNKNOWN"); bearing=e["aoa_deg"]%360
            offset=-5.0 if 0<=bearing<180 else 5.0
            ax.text(a+math.radians(offset),r,lib,ha="right" if offset<0 else "left",va="center",fontsize=5.5,alpha=.78,zorder=4)
        self.draw_idle()


class EnhancedS2BOperatorWindow(S2BOperatorWindow):
    """Enhanced UI fed only by measured PDWs from the unified 40 MS/s stream."""

    BLOCK_SAMPLES=40_000                 # 1 ms blocks
    SIM_SECONDS_PER_REFRESH=0.100        # 100 x 1 ms blocks per UI refresh

    def __init__(self):
        self.emitter_library=EmitterLibrary()
        self.mode_history=ObservedModeHistory(max_entries=10)
        self.library_memory={}; self.library_degraded=set(); self.library_operator_confirmed=set()
        self.behaviour=IlluminationBehaviourManager(
            illumination_threshold_db=-8.0, persistent_s=1.0,
            peak_separation_s=.25, baseline_confidence_threshold=.50,
            change_confidence_threshold=.50, change_hold_s=5.0,
        )
        self.stream_source=None; self.stream_processor=None; self.stream_tracker=None
        self.last_stream_time_s=0.0; self._details_emitter_id=None
        super().__init__()
        self.setWindowTitle("S2B ESM - Unified 40 MS/s Operator Display")
        self.resize(1750,1050); self.setMinimumSize(1250,760); self.details.setMinimumHeight(380)
        # The base timer is UI cadence only. Each tick processes a fixed amount of
        # simulated 40 MS/s stream; later an Ettus source will supply blocks continuously.
        self.timer.setInterval(100)
        self._show_prestart_blank()

    def _build_ui(self):
        super()._build_ui()
        old=self.polar; parent=old.parentWidget(); layout=parent.layout(); index=layout.indexOf(old)
        layout.removeWidget(old); old.setParent(None); self.polar=EnhancedPolarEmitterCanvas(parent); layout.insertWidget(index,self.polar,stretch=1)
        self.emitter_table.setColumnCount(8); self.emitter_table.setHorizontalHeaderLabels(["Emitter","Library","AOA","RF MHz","Waveform","State","Watch","Track conf."])
        self.mode_history_title=QLabel("RECENT OBSERVED MODES (1 s cells)"); self.mode_history_title.setStyleSheet("font-weight:700; margin-top:2px;")
        self.mode_history_table=QTableWidget(1,10); self.mode_history_table.setVerticalHeaderLabels(["MODE"]); self.mode_history_table.horizontalHeader().setVisible(False); self.mode_history_table.setFixedHeight(58); self.mode_history_table.setSelectionMode(QTableWidget.NoSelection); self.mode_history_table.setFocusPolicy(Qt.NoFocus)
        table_index=layout.indexOf(self.emitter_table); layout.insertWidget(table_index,self.mode_history_title); layout.insertWidget(table_index+1,self.mode_history_table)
        self.confirm_library_button=QPushButton("CONFIRM LIBRARY ID"); self.confirm_library_button.clicked.connect(self._confirm_library_id); layout.insertWidget(table_index+2,self.confirm_library_button)

    def _new_stream(self):
        self.stream_source=SimulatedStreamingIQSource(
            sample_rate_hz=40_000_000, center_frequency_hz=9_400_000_000,
            block_samples=self.BLOCK_SAMPLES, noise_std=.02,
        )
        self.stream_processor=StreamingESMProcessor(self.stream_source)
        self.stream_tracker=StreamingEmitterTracker(frequency_gate_hz=2_000_000.0)
        self.behaviour=IlluminationBehaviourManager(
            illumination_threshold_db=-8.0, persistent_s=1.0,
            peak_separation_s=.25, baseline_confidence_threshold=.50,
            change_confidence_threshold=.50, change_hold_s=5.0,
        )
        self.last_stream_time_s=0.0

    def start_system(self):
        if self.running:return
        if self.stream_processor is None:self._new_stream()
        self.running=True; self._set_status("RUNNING"); self.timer.start(); self._refresh()

    def stop_system(self):
        if not self.running:return
        self.running=False; self.timer.stop(); self._set_status("STOPPED")

    def reset_system(self):
        self.timer.stop(); self.running=False
        self.mode_history.clear(); self.library_memory.clear(); self.library_degraded.clear(); self.library_operator_confirmed.clear()
        self.watched_emitters.clear(); self.operator_assessments.clear(); self._details_emitter_id=None
        self._new_stream(); self._show_prestart_blank(); self._set_status("STOPPED")

    def _scenario_changed(self,_name):
        # The unified streaming source owns its physical scenario. The old finite-IQ
        # scenario selector is retained visually for now but cannot inject emitters.
        pass

    def _show_prestart_blank(self):
        self.emitters=[]; self.selected_emitter_index=0; self.emitter_table.setRowCount(0); self.polar.update_emitters([],0)
        if hasattr(self,"mode_history_table"):self.mode_history_table.clearContents(); self.mode_history_title.setText("RECENT OBSERVED MODES - WAITING FOR START")
        if hasattr(self,"confirm_library_button"):self.confirm_library_button.setEnabled(False)
        self.emitter_heading.setText("NO EMITTER SELECTED"); self.assessment_label.setText("UNASSESSED"); self.assessment_label.setStyleSheet("background:#777;color:white;font-weight:700;padding:7px;")
        self.watch_button.setEnabled(False); self.details.setPlainText("Press START to begin unified 40 MS/s IQ processing.\n\nNo emitter is inserted from scenario truth.")
        self.statusBar().showMessage("Stopped - operator picture blank until measured PDWs are acquired")

    @staticmethod
    def _display_aoa(track):
        # AOA estimation is not yet implemented in the single-channel 40 MS/s front
        # end. Keep the existing simulated AOA observation for E1/E2 and use the
        # known test-sector bearing for the 9.410 GHz candidate only as a temporary
        # sensor stub. This does not affect detection, classification, PRI or behaviour.
        return 135.0 if abs(track.frequency_hz-9_410_000_000)<=2_000_000 else 45.0

    def _feed_behaviour(self, track, new_pdws, now_s):
        # Convert sparse measured pulse amplitudes into 10 ms observation bins. Empty
        # bins are below illumination threshold. The tracker therefore sees only the
        # measured PDW stream, never beam azimuth, scripted received power or mode.
        relevant=[p for p in new_pdws if abs(p.frequency_hz-track.frequency_hz)<=self.stream_tracker.frequency_gate_hz]
        bin_s=.010; start=max(0.0,self.last_stream_time_s); end=now_s
        by_bin={}
        for p in relevant:
            b=int(p.toa_s/bin_s); by_bin[b]=max(by_bin.get(b,-120.0),p.amplitude_dbfs)
        assessment=self.behaviour.assessment(track.emitter_id, start)
        b0=int(start/bin_s); b1=int(math.ceil(end/bin_s))
        for b in range(b0,b1):
            t=(b+1)*bin_s; amp=by_bin.get(b,-120.0); assessment=self.behaviour.update(track.emitter_id,t,amp)
        return assessment

    def _assign_library(self,e):
        c=e["current"]; illumination=e.get("illumination"); state=illumination.state if illumination is not None else None; eid=e["emitter_id"]
        match=self.emitter_library.identify(c.get("frequency_hz"),c.get("pri_s"),c.get("modulation"),state,previous_type=self.library_memory.get(eid))
        if eid in self.library_operator_confirmed: lib_id="NAVRADAR"; conf=1.0; reason="NAVRADAR identity confirmed by operator"
        elif eid in self.library_degraded:
            lib_id="NAVRADAR?"; conf=.40; reason="Prior contradictory measured behaviour remains unresolved; operator confirmation required"
        elif match.emitter_type=="NAVRADAR?":
            self.library_degraded.add(eid); lib_id="NAVRADAR?"; conf=.40; reason="Measured behaviour contradicts the earlier NAVRADAR library match; operator confirmation required"
        else: lib_id=match.emitter_type; conf=match.confidence; reason=match.reason
        e["library_id"]=lib_id; e["library_confidence"]=conf; e["library_reason"]=reason; self.library_memory[eid]=lib_id; return e

    def _refresh(self):
        if not self.running:return
        try:
            blocks=max(1,int(round(self.SIM_SECONDS_PER_REFRESH/(self.BLOCK_SAMPLES/40_000_000.0))))
            new_pdws,last_meta=self.stream_processor.process_blocks(blocks)
            self.stream_tracker.update(new_pdws)
            now_s=self.stream_source.time_s
            out=[]
            for track in self.stream_tracker.tracks:
                current=track.summary(); illum=self._feed_behaviour(track,new_pdws,now_s)
                system_state=illum.system_assessment if illum.state!="UNRESOLVED" else ("MONITOR" if track.total_pulses>=3 else "UNASSESSED")
                eid=track.emitter_id
                e={"emitter_id":eid,"aoa_deg":self._display_aoa(track),"system_state":system_state,"state":system_state,"display_color":ASSESSMENT_COLORS.get(system_state,ASSESSMENT_COLORS["UNASSESSED"]),"watched":eid in self.watched_emitters,"tracks":[],"current":current,"links":[],"track_confidence":min(1.0,track.total_pulses/20.0),"illumination":illum,"last_seen_s":track.last_seen_s}
                self._apply_operator_assessment(e); self._assign_library(e); out.append(e)
                if illum.state in ("PERIODIC_SCAN","PERSISTENT_ILLUMINATION"): self.mode_history.update(eid,max(0.,now_s-.010),illum.state)
            self.emitters=out
            if self.selected_emitter_index>=len(out):self.selected_emitter_index=max(0,len(out)-1)
            self._populate_table(); self.polar.update_emitters(out,self.selected_emitter_index); self._show_selected_emitter(); self._update_mode_history_display()
            self.last_stream_time_s=now_s
            self.statusBar().showMessage(f"Unified sampled-IQ stream: {now_s:6.2f} s | 40.0 MS/s | PDWs {self.stream_processor.total_pdws} | persistent emitters {len(out)}")
        except Exception as exc:
            self.timer.stop(); self.running=False; self._set_status("ERROR"); self.details.setPlainText(f"Streaming UI error:\n\n{exc}"); self.statusBar().showMessage(str(exc))

    def _confirm_library_id(self):
        if not self.emitters:return
        e=self.emitters[self.selected_emitter_index]; eid=e["emitter_id"]
        if e.get("library_id") not in ("NAVRADAR","NAVRADAR?"):return
        self.library_operator_confirmed.add(eid); self.library_degraded.discard(eid); self._assign_library(e); self._populate_table(); self.polar.update_emitters(self.emitters,self.selected_emitter_index); self._show_selected_emitter(); self._update_mode_history_display()

    def _populate_table(self):
        self.emitter_table.setRowCount(len(self.emitters))
        for row,e in enumerate(self.emitters):
            c=e["current"]; vals=(e["emitter_id"],e.get("library_id","UNKNOWN"),f"{e['aoa_deg']:.1f} deg",f"{c['frequency_hz']/1e6:.3f}",c["modulation"],e["state"],"WATCH" if e.get("watched",False) else "",f"{100*e['track_confidence']:.1f}%")
            for col,value in enumerate(vals):
                item=QTableWidgetItem(value); item.setFlags(item.flags() & ~Qt.ItemIsEditable); self.emitter_table.setItem(row,col,item)
        if self.emitters:self.emitter_table.selectRow(self.selected_emitter_index)

    def _emitter_selected(self,row,column):
        if self.emitters and 0<=row<len(self.emitters) and self.emitters[row]["emitter_id"]!=self._details_emitter_id:self._details_emitter_id=None
        self.selected_emitter_index=row; self.polar.update_emitters(self.emitters,row); self._show_selected_emitter(); self._update_mode_history_display()

    def _update_mode_history_display(self):
        self.mode_history_table.clearContents()
        if not self.emitters:
            self.mode_history_title.setText("RECENT OBSERVED MODES - NO EMITTER SELECTED"); self.confirm_library_button.setEnabled(False); return
        e=self.emitters[self.selected_emitter_index]; labels=self.mode_history.labels(e["emitter_id"]); padded=[""]*(10-len(labels))+labels[-10:]
        for col,label in enumerate(padded):
            item=QTableWidgetItem(label); item.setTextAlignment(Qt.AlignCenter); item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if label=="SEARCH":item.setBackground(Qt.lightGray)
            elif label=="DWELL":item.setBackground(Qt.yellow)
            self.mode_history_table.setItem(0,col,item)
        self.mode_history_title.setText(f"{e['emitter_id']} / {e.get('library_id','UNKNOWN')} - RECENT OBSERVED MODES (measured PDWs, newest at right)")
        provisional=e.get("library_id")=="NAVRADAR?"; self.confirm_library_button.setEnabled(provisional); self.confirm_library_button.setText("CONFIRM NAVRADAR ID" if provisional else "LIBRARY ID CONFIRMED" if e["emitter_id"] in self.library_operator_confirmed else "CONFIRM LIBRARY ID")

    def _show_selected_emitter(self):
        if not self.emitters:
            self._show_prestart_blank(); return
        e=self.emitters[self.selected_emitter_index]; eid=e["emitter_id"]; c=e["current"]; illum=e["illumination"]
        scrollbar=self.details.verticalScrollBar(); same=eid==self._details_emitter_id; old_scroll=scrollbar.value() if same else 0
        self.emitter_heading.setText(f"{eid}  |  BEARING {e['aoa_deg']:.1f} deg"+("  * WATCH" if e.get("watched") else "")); self.assessment_label.setText(e["state"]); self.assessment_label.setStyleSheet(f"background:{e['display_color']};color:white;font-weight:700;padding:7px;"); self.watch_button.setEnabled(True); self.watch_button.setText("REMOVE WATCH" if e.get("watched") else "WATCH SELECTED EMITTER")
        pri="UNRESOLVED" if c["pri_s"] is None else f"{c['pri_s']*1e6:.1f} us"
        period="-" if illum.scan_period_s is None else f"{illum.scan_period_s:.3f} s"; rate="-" if illum.scan_rate_rpm is None else f"{illum.scan_rate_rpm:.1f} RPM"
        recent="-" if illum.recent_change_from is None else f"{illum.recent_change_from} -> {illum.recent_change_to}"
        text=("EMITTER LIBRARY\n---------------\n"f"ID / confidence  : {e.get('library_id','UNKNOWN')} / {100*e.get('library_confidence',0):.0f}%\n"f"Evidence         : {e.get('library_reason','')}\n\nCURRENT MEASURED STATE\n----------------------\n"f"Physical track   : {eid}\nBearing          : {e['aoa_deg']:.1f} deg (AOA sensor stub)\nRF               : {c['frequency_hz']/1e6:.3f} MHz\nPRI median       : {pri}\nPulse width      : {c['pulse_width_s']*1e6:.3f} us\nWaveform family  : {c['modulation']}\nPeak level       : {c['amplitude_dbfs']:.2f} dBFS\nMeasured pulses  : {c['pulse_count']}\nLast detection   : {e['last_seen_s']:.3f} s\n\nBEHAVIOUR FROM MEASURED PDWs\n----------------------------\n"f"Current behaviour: {illum.state}\nEvidence         : {100*illum.confidence:.1f}%\nBaseline         : {illum.baseline_state or '-'}\nCurrent period   : {period}\nEstimated rate   : {rate}\nRecent change    : {recent}\n\nS2B INTERPRETATION\n------------------\nNo scripted E3 observation path exists. Detection, RF, PW, waveform, amplitude and illumination behaviour above originate from the common sampled-IQ/PDW chain.")
        self.details.setPlainText(text); scrollbar.setValue(min(old_scroll,scrollbar.maximum()) if same else 0); self._details_emitter_id=eid


def main():
    app=QApplication(sys.argv); app.setApplicationName("S2B ESM"); window=EnhancedS2BOperatorWindow(); window.show(); sys.exit(app.exec_())


if __name__=="__main__":main()
