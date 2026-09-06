import sys
import math
import random
import time

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
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
from beam_model import RotatingSincBeam
from illumination_behaviour import EmitterIlluminationTracker

REFRESH_MS = 750
ASSESSMENT_COLORS = {"UNASSESSED":"#777777","MONITOR":"#2f7fbf","CHANGED":"#d28b00","OF INTEREST":"#e56b00","THREAT":"#b22222"}

class PolarEmitterCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(6.2,6.2), tight_layout=True); self.axes=self.figure.add_subplot(111,projection="polar")
        super().__init__(self.figure); self.setParent(parent); self._configure_axes()
    def _configure_axes(self):
        ax=self.axes; ax.clear(); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_ylim(0,1); ax.set_yticks((.25,.5,.75,1)); ax.set_yticklabels(()); ax.set_thetagrids(range(0,360,45)); ax.grid(True,alpha=.28); ax.set_title("EMITTER BEARING PICTURE",pad=18,fontsize=12,fontweight="bold")
        for b in range(0,360,45):
            a=math.radians(b); ax.plot([a,a],[0,1],color="black",linewidth=.7,alpha=.16,zorder=0)
        items=[Line2D([0],[0],marker="o",linestyle="None",markerfacecolor=ASSESSMENT_COLORS[s],markeredgecolor="black",markersize=8,label=s) for s in ("UNASSESSED","MONITOR","CHANGED","OF INTEREST","THREAT")]
        ax.legend(handles=items,title="ASSESSMENT",loc="upper right",bbox_to_anchor=(-.18,1.08),framealpha=.92,fontsize=8,title_fontsize=8,borderaxespad=0)
    def update_emitters(self, emitters, selected_index=0):
        self._configure_axes(); ax=self.axes
        if not emitters: ax.text(.5,.5,"NO EMITTERS",transform=ax.transAxes,ha="center",va="center",fontsize=12); self.draw_idle(); return
        counts={}; seen={}
        for e in emitters: k=round(e["aoa_deg"],1); counts[k]=counts.get(k,0)+1
        for i,e in enumerate(emitters):
            a=math.radians(e["aoa_deg"]%360); k=round(e["aoa_deg"],1); n=seen.get(k,0); seen[k]=n+1; r=.78 if counts[k]==1 else .60+.18*n
            ax.plot([a,a],[0,r],color="black",linewidth=1.5 if i==selected_index else 1,alpha=.75 if i==selected_index else .55,zorder=1)
            size=125 if i==selected_index else 90; marker="D" if i==selected_index else "o"
            ax.scatter([a],[r],s=size,marker=marker,c=[e["display_color"]],edgecolors="black",linewidths=.8,zorder=3)
            if e.get("watched",False): ax.scatter([a],[r],s=size+115,marker="o",facecolors="none",edgecolors="#d4a017",linewidths=2.3,zorder=2)
            ax.text(a,min(r+.09,.98),e["emitter_id"]+(" *" if e.get("watched",False) else ""),ha="center",va="center",fontsize=10,fontweight="bold",zorder=4)
        self.draw_idle()

class S2BOperatorWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("S2B ESM - Operator Display"); self.resize(1450,880)
        self.running=False; self.emitters=[]; self.selected_emitter_index=0; self.extractor=None; self.watched_emitters=set()
        self.nav_elapsed_s=0.; self.nav_resume_monotonic=None; self.nav_last_sample_s=0.
        self.nav_beam=RotatingSincBeam(beamwidth_deg=3.,scan_rate_rpm=30.,initial_azimuth_deg=45.,sidelobe_floor_db=-50.)
        self.nav_tracker=self._new_nav_tracker(); self.nav_assessment=self.nav_tracker.assess(0.); self.nav_level_dbfs=-55.
        self.timer=QTimer(self); self.timer.setInterval(REFRESH_MS); self.timer.timeout.connect(self._refresh)
        self._build_ui(); self._set_status("STOPPED"); self._refresh()
    def _new_nav_tracker(self):
        return EmitterIlluminationTracker(history_s=30.,illumination_threshold_db=-18.,persistent_s=.75,peak_separation_s=.5,baseline_confidence_threshold=.50,change_confidence_threshold=.50,change_hold_s=3.)
    def _build_ui(self):
        central=QWidget(self); self.setCentralWidget(central); root=QVBoxLayout(central); top=QHBoxLayout(); title=QLabel("S2B ESM OPERATOR DISPLAY"); title.setStyleSheet("font-size: 18px; font-weight: 700;"); top.addWidget(title); top.addStretch(1); top.addWidget(QLabel("Scenario:"))
        self.scenario_combo=QComboBox(); [self.scenario_combo.addItem(s.name) for s in list_scenarios()]; idx=self.scenario_combo.findText("close_emitters"); self.scenario_combo.setCurrentIndex(idx if idx>=0 else 0); self.scenario_combo.currentTextChanged.connect(self._scenario_changed); top.addWidget(self.scenario_combo)
        self.status_label=QLabel(); self.status_label.setMinimumWidth(100); self.status_label.setAlignment(Qt.AlignCenter); top.addWidget(self.status_label)
        for text,fn in (("START",self.start_system),("STOP",self.stop_system),("RESET",self.reset_system),("EXIT",self.close)):
            b=QPushButton(text); b.clicked.connect(fn); top.addWidget(b)
        root.addLayout(top); splitter=QSplitter(Qt.Horizontal); root.addWidget(splitter,stretch=1)
        left=QWidget(); ll=QVBoxLayout(left); self.polar=PolarEmitterCanvas(left); ll.addWidget(self.polar,stretch=1)
        self.emitter_table=QTableWidget(0,7); self.emitter_table.setHorizontalHeaderLabels(["Emitter","AOA","RF MHz","Waveform","State","Watch","Track conf."]); self.emitter_table.setSelectionBehavior(QTableWidget.SelectRows); self.emitter_table.setSelectionMode(QTableWidget.SingleSelection); self.emitter_table.cellClicked.connect(self._emitter_selected); self.emitter_table.horizontalHeader().setStretchLastSection(True); self.emitter_table.setMaximumHeight(230); ll.addWidget(self.emitter_table); splitter.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); self.emitter_heading=QLabel("NO EMITTER SELECTED"); self.emitter_heading.setStyleSheet("font-size: 17px; font-weight: 700;"); rl.addWidget(self.emitter_heading)
        self.assessment_label=QLabel("UNASSESSED"); self.assessment_label.setAlignment(Qt.AlignCenter); self.assessment_label.setMinimumHeight(38); rl.addWidget(self.assessment_label)
        self.watch_button=QPushButton("WATCH SELECTED EMITTER"); self.watch_button.clicked.connect(self._toggle_watch); self.watch_button.setEnabled(False); rl.addWidget(self.watch_button)
        self.details=QTextEdit(); self.details.setReadOnly(True); self.details.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 12px;"); rl.addWidget(self.details,stretch=3)
        nt=QLabel("OPERATOR / S2B NOTES"); nt.setStyleSheet("font-weight: 700;"); rl.addWidget(nt); self.notes=QTextEdit(); self.notes.setPlaceholderText("Operator notes. Later this panel will also show behavioural hypotheses, missing evidence and suggested probes."); rl.addWidget(self.notes,stretch=1); splitter.addWidget(right); splitter.setSizes([880,570])
        footer=QLabel("START resumes. STOP pauses. RESET clears emitter behaviour history and restarts the scenario. Colour = system assessment; gold ring/* = operator WATCH. Radial position is not range."); footer.setStyleSheet("color: #666;"); root.addWidget(footer)
    def _set_status(self,state):
        self.status_label.setText(state); bg="#1f7a3a" if state=="RUNNING" else "#9b1c1c" if state=="ERROR" else "#555"; self.status_label.setStyleSheet(f"background:{bg};color:white;font-weight:700;padding:6px;")
    def _reset_nav_radar(self):
        self.nav_elapsed_s=0.; self.nav_resume_monotonic=time.monotonic() if self.running else None; self.nav_last_sample_s=0.; self.nav_tracker=self._new_nav_tracker(); self.nav_assessment=self.nav_tracker.assess(0.); self.nav_level_dbfs=-55.
    def start_system(self):
        if self.running:return
        self.running=True; self.nav_resume_monotonic=time.monotonic(); self._set_status("RUNNING"); self.timer.start(); self._refresh()
    def stop_system(self):
        if not self.running:return
        if self.nav_resume_monotonic is not None:self.nav_elapsed_s+=time.monotonic()-self.nav_resume_monotonic
        self.nav_resume_monotonic=None; self.running=False; self.timer.stop(); self._set_status("STOPPED")
    def reset_system(self):
        was=self.running; self.extractor=None; self.selected_emitter_index=0; self.watched_emitters.clear(); self._reset_nav_radar()
        if was:self.nav_resume_monotonic=time.monotonic(); self.timer.start(); self._set_status("RUNNING")
        else:self._set_status("STOPPED")
        self._refresh()
    def closeEvent(self,event): self.timer.stop(); self.running=False; event.accept()
    def _scenario_changed(self,_name): self.selected_emitter_index=0; self.extractor=None; self.watched_emitters.clear(); self._reset_nav_radar(); self._refresh()
    def _make_mht(self):
        return GatedFastProbabilisticMultipleHypothesisAssociator(frequency_sigma_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,pulse_width_sigma_s=ASSOCIATION_PULSE_WIDTH_TOLERANCE_S,amplitude_sigma_db=ASSOCIATION_AMPLITUDE_TOLERANCE_DB,timing_sigma_s=ASSOCIATION_TIMING_TOLERANCE_S,beam_width=MHT_BEAM_WIDTH,max_emitters=MHT_MAX_EMITTERS,birth_probability=PMHT_BIRTH_PROBABILITY,clutter_probability=PMHT_CLUTTER_PROBABILITY,modulation_match_probability=PMHT_MODULATION_MATCH_PROBABILITY,missed_pulse_probability=PMHT_MISSED_PULSE_PROBABILITY,max_pri_multiple=PMHT_MAX_PRI_MULTIPLE)
    def _process_snapshot(self):
        scenario=get_scenario(self.scenario_combo.currentText()); source=SimulatedSource(sample_rate_hz=SAMPLE_RATE_HZ,center_frequency_hz=CENTER_FREQUENCY_HZ,emitters=scenario.emitters,noise_std=scenario.noise_std); iq,metadata=source.read(); detector=PulseDetector(threshold=DETECTION_THRESHOLD,sample_rate_hz=metadata["sample_rate_hz"],min_pulse_width_s=MIN_PULSE_WIDTH_S)
        if self.extractor is None:self.extractor=PDWExtractor(sample_rate_hz=metadata["sample_rate_hz"],center_frequency_hz=metadata["center_frequency_hz"])
        else:self.extractor.next_pdw_id=1
        pulses=detector.detect(iq); pdws=[self.extractor.extract(iq,p) for p in pulses]; mht=self._make_mht(); hypotheses=mht.associate(pdws)
        if not hypotheses:return scenario,[],len(pulses)
        marg=mht.association_marginals(hypotheses); mem=mht.reference_track_membership(hypotheses); summaries=OperatorEmitterSummary().build(hypotheses,marg,mem); ph=PhysicalEmitterCorrelator(frequency_scale_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,amplitude_scale_db=ASSOCIATION_AMPLITUDE_TOLERANCE_DB).correlate(hypotheses[0],[]); groups=_physical_groups(summaries,ph); emitters=[]
        for ei,g in enumerate(groups,start=1):
            tracks=g["tracks"]; current=max(tracks,key=lambda x:x["end_toa_s"]); changed=len(tracks)>1; aoa=sum(t.get("aoa_deg",0.) for t in tracks)/len(tracks); conf=sum(t["track_confidence"] for t in tracks)/len(tracks); pc=sum(t["pulse_count"] for t in tracks); state="UNASSESSED" if pc<3 or conf<.75 else "CHANGED" if changed else "MONITOR"; eid=f"E{ei}"
            emitters.append({"emitter_id":eid,"aoa_deg":aoa,"state":state,"display_color":ASSESSMENT_COLORS[state],"watched":eid in self.watched_emitters,"tracks":tracks,"current":current,"links":g["links"],"track_confidence":conf,"illumination":None})
        return scenario,emitters,len(pulses)
    def _update_nav_radar(self):
        if not self.running:return
        delta=time.monotonic()-self.nav_resume_monotonic if self.nav_resume_monotonic is not None else 0.; now=self.nav_elapsed_s+delta; t=self.nav_last_sample_s
        while t<=now+1e-9:
            gain=self.nav_beam.gain_db(135.,t); self.nav_level_dbfs=max(-55.,-6.+gain)+random.gauss(0,.6); self.nav_assessment=self.nav_tracker.update(t,self.nav_level_dbfs); t+=.01
        self.nav_last_sample_s=max(self.nav_last_sample_s,t)
    def _nav_emitter_record(self):
        a=self.nav_assessment; state=a.system_assessment; eid="E3"; current={"frequency_hz":9.410e9,"modulation":"CW","pri_s":1e-3,"pri_pattern":"STABLE","pulse_width_s":5e-6,"amplitude_dbfs":self.nav_level_dbfs,"pulse_count":max(0,int(self.nav_last_sample_s/.001)),"track_id":3,"end_toa_s":self.nav_last_sample_s}
        return {"emitter_id":eid,"aoa_deg":135.,"state":state,"display_color":ASSESSMENT_COLORS.get(state,ASSESSMENT_COLORS["UNASSESSED"]),"watched":eid in self.watched_emitters,"tracks":[current],"current":current,"links":[],"track_confidence":max(0.,a.confidence),"illumination":a}
    def _refresh(self):
        try:
            scenario,emitters,pulse_count=self._process_snapshot(); self._update_nav_radar(); emitters.append(self._nav_emitter_record()); self.emitters=emitters
            if self.selected_emitter_index>=len(emitters):self.selected_emitter_index=max(0,len(emitters)-1)
            self._populate_table(); self.polar.update_emitters(self.emitters,self.selected_emitter_index); self._show_selected_emitter(); self.statusBar().showMessage(f"Scenario: {scenario.name} | IQ pulses: {pulse_count} | Physical emitters: {len(emitters)} | E3: 9.410 GHz nav radar")
        except Exception as exc:
            self.timer.stop(); self.running=False; self._set_status("ERROR"); self.statusBar().showMessage(str(exc)); self.details.setPlainText(f"UI processing error:\n\n{exc}")
    def _populate_table(self):
        self.emitter_table.setRowCount(len(self.emitters))
        for row,e in enumerate(self.emitters):
            c=e["current"]; vals=(e["emitter_id"],f"{e['aoa_deg']:.1f} deg",f"{c['frequency_hz']/1e6:.3f}",c["modulation"],e["state"],"WATCH" if e.get("watched",False) else "",f"{100*e['track_confidence']:.1f}%")
            for col,v in enumerate(vals): item=QTableWidgetItem(v); item.setFlags(item.flags() & ~Qt.ItemIsEditable); self.emitter_table.setItem(row,col,item)
        if self.emitters:self.emitter_table.selectRow(self.selected_emitter_index)
    def _emitter_selected(self,row,_column): self.selected_emitter_index=row; self.polar.update_emitters(self.emitters,row); self._show_selected_emitter()
    def _toggle_watch(self):
        if not self.emitters:return
        eid=self.emitters[self.selected_emitter_index]["emitter_id"]
        if eid in self.watched_emitters:self.watched_emitters.remove(eid)
        else:self.watched_emitters.add(eid)
        for e in self.emitters:e["watched"]=e["emitter_id"] in self.watched_emitters
        self._populate_table(); self.polar.update_emitters(self.emitters,self.selected_emitter_index); self._show_selected_emitter()
    def _show_selected_emitter(self):
        if not self.emitters:
            self.emitter_heading.setText("NO EMITTER SELECTED"); self.assessment_label.setText("UNASSESSED"); self.assessment_label.setStyleSheet("background:#777;color:white;font-weight:700;padding:7px;"); self.watch_button.setEnabled(False); self.details.setPlainText("No physical emitters currently assessed."); return
        e=self.emitters[self.selected_emitter_index]; c=e["current"]; watched=e.get("watched",False); self.emitter_heading.setText(f"{e['emitter_id']}  |  BEARING {e['aoa_deg']:.1f} deg"+("  * WATCH" if watched else "")); self.assessment_label.setText(e["state"]); self.assessment_label.setStyleSheet(f"background:{e['display_color']};color:white;font-weight:700;padding:7px;"); self.watch_button.setEnabled(True); self.watch_button.setText("REMOVE WATCH" if watched else "WATCH SELECTED EMITTER")
        pri="UNRESOLVED" if c["pri_s"] is None else f"{c['pri_s']*1e6:.1f} us"; tids=", ".join(f"T{t['track_id']}" for t in e["tracks"])
        lines=["CURRENT OBSERVED STATE","----------------------",f"Physical emitter : {e['emitter_id']}",f"Bearing          : {e['aoa_deg']:.1f} deg",f"Operator watch   : {'YES' if watched else 'NO'}",f"System assessment: {e['state']}",f"Sequence tracks  : {tids}",f"RF               : {c['frequency_hz']/1e6:.3f} MHz",f"PRI median       : {pri}",f"PRI pattern      : {c['pri_pattern']}",f"Pulse width      : {c['pulse_width_s']*1e6:.3f} us",f"Waveform family  : {c['modulation']}",f"Level            : {c['amplitude_dbfs']:.2f} dBFS",f"Pulses           : {c['pulse_count']}",f"Track confidence : {100*e['track_confidence']:.1f}%","","BEHAVIOUR / CHANGE","------------------"]
        a=e.get("illumination")
        if a is not None:
            period="UNRESOLVED" if a.scan_period_s is None else f"{a.scan_period_s:.3f} s"; rpm="UNRESOLVED" if a.scan_rate_rpm is None else f"{a.scan_rate_rpm:.1f} RPM"; lines.extend([f"Illumination     : {a.state}",f"Scan period      : {period}",f"Estimated rate   : {rpm}",f"Period evidence  : {100*a.confidence:.1f}%",f"Baseline         : {a.baseline_state or 'NOT ESTABLISHED'}"])
            if a.recent_change_from and a.recent_change_to:
                age=max(0.,self.nav_last_sample_s-(a.recent_change_time_s or self.nav_last_sample_s)); lines.append(f"Recent change    : {a.recent_change_from} -> {a.recent_change_to} ({age:.1f} s ago)")
        elif len(e["tracks"])>1:
            lines.append("Observable state change detected between linked sequence tracks.")
            for t in e["tracks"]:
                tp="UNRESOLVED" if t["pri_s"] is None else f"{t['pri_s']*1e6:.1f} us"; lines.append(f"T{t['track_id']}: RF={t['frequency_hz']/1e6:.3f} MHz, PRI={tp}, PW={t['pulse_width_s']*1e6:.3f} us, MOD={t['modulation']}")
        else: lines.append("No significant linked-track behaviour change currently detected.")
        lines.extend(["","S2B INTERPRETATION","------------------","Behaviour hypotheses are not enabled yet.","OF INTEREST and THREAT are therefore not assigned automatically yet.","WATCH is an operator attention flag and does not change system assessment."]); self.details.setPlainText("\n".join(lines))

def main():
    app=QApplication(sys.argv); app.setApplicationName("S2B ESM"); window=S2BOperatorWindow(); window.show(); sys.exit(app.exec_())
if __name__=="__main__": main()
