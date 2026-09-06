import sys
import math

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QTableWidget, QTableWidgetItem

from emitter_library import EmitterLibrary
from mode_history import ObservedModeHistory
from esm_operator_ui import S2BOperatorWindow, PolarEmitterCanvas


class EnhancedPolarEmitterCanvas(PolarEmitterCanvas):
    def update_emitters(self, emitters, selected_index=0):
        super().update_emitters(emitters, selected_index)
        ax = self.axes
        counts = {}; seen = {}
        for e in emitters:
            k=round(e["aoa_deg"],1); counts[k]=counts.get(k,0)+1
        for e in emitters:
            a=math.radians(e["aoa_deg"]%360); k=round(e["aoa_deg"],1); n=seen.get(k,0); seen[k]=n+1
            r=.78 if counts[k]==1 else .60+.18*n; lib=e.get("library_id","UNKNOWN")
            bearing=e["aoa_deg"]%360; offset_deg=-5.0 if 0<=bearing<180 else 5.0
            ax.text(a+math.radians(offset_deg),r,lib,ha="right" if offset_deg<0 else "left",va="center",fontsize=5.5,alpha=.78,fontweight="normal",zorder=4)
        self.draw_idle()


class EnhancedS2BOperatorWindow(S2BOperatorWindow):
    def __init__(self):
        self.emitter_library=EmitterLibrary()
        # A short strip is deliberately a recent-context display, not a long log.
        self.mode_history=ObservedModeHistory(max_entries=10)
        self.library_memory={}
        self._details_emitter_id=None
        super().__init__()
        self.setWindowTitle("S2B ESM - Operator Display"); self.resize(1750,1050); self.setMinimumSize(1250,760); self.details.setMinimumHeight(380)

    def _build_ui(self):
        super()._build_ui(); old=self.polar; parent=old.parentWidget(); layout=parent.layout(); index=layout.indexOf(old); layout.removeWidget(old); old.setParent(None); self.polar=EnhancedPolarEmitterCanvas(parent); layout.insertWidget(index,self.polar,stretch=1)
        self.emitter_table.setColumnCount(8); self.emitter_table.setHorizontalHeaderLabels(["Emitter","Library","AOA","RF MHz","Waveform","State","Watch","Track conf."])
        self.mode_history_title=QLabel("RECENT OBSERVED MODES (1 s cells)"); self.mode_history_title.setStyleSheet("font-weight:700; margin-top:2px;")
        self.mode_history_table=QTableWidget(1,10); self.mode_history_table.setVerticalHeaderLabels(["MODE"]); self.mode_history_table.horizontalHeader().setVisible(False); self.mode_history_table.setFixedHeight(58); self.mode_history_table.setSelectionMode(QTableWidget.NoSelection); self.mode_history_table.setFocusPolicy(Qt.NoFocus)
        table_index=layout.indexOf(self.emitter_table); layout.insertWidget(table_index,self.mode_history_title); layout.insertWidget(table_index+1,self.mode_history_table)

    def reset_system(self):
        self.mode_history.clear(); self.library_memory.clear(); self._details_emitter_id=None; super().reset_system()

    def _scenario_changed(self,name):
        self.mode_history.clear(); self.library_memory.clear(); self._details_emitter_id=None; super()._scenario_changed(name)

    def _assign_library(self,e):
        c=e["current"]; illumination=e.get("illumination"); illumination_state=illumination.state if illumination is not None else None; eid=e["emitter_id"]
        previous=self.library_memory.get(eid)
        match=self.emitter_library.identify(c.get("frequency_hz"),c.get("pri_s"),c.get("modulation"),illumination_state,previous_type=previous)
        e["library_id"]=match.emitter_type; e["library_confidence"]=match.confidence; e["library_reason"]=match.reason
        if match.emitter_type != "UNKNOWN": self.library_memory[eid]=match.emitter_type
        return e

    def _refresh(self):
        super()._refresh()
        if not self.emitters:return
        for e in self.emitters:self._assign_library(e)
        e3=next((e for e in self.emitters if e["emitter_id"]=="E3"),None)
        if e3 is not None and e3.get("illumination") is not None:
            self.mode_history.update("E3",max(0.,self.nav_last_sample_s-.010),e3["illumination"].state)
        self._populate_table(); self.polar.update_emitters(self.emitters,self.selected_emitter_index); self._show_selected_emitter(); self._update_mode_history_display()

    def _populate_table(self):
        self.emitter_table.setRowCount(len(self.emitters))
        for row,e in enumerate(self.emitters):
            c=e["current"]; vals=(e["emitter_id"],e.get("library_id","UNKNOWN"),f"{e['aoa_deg']:.1f} deg",f"{c['frequency_hz']/1e6:.3f}",c["modulation"],e["state"],"WATCH" if e.get("watched",False) else "",f"{100*e['track_confidence']:.1f}%")
            for col,value in enumerate(vals):
                item=QTableWidgetItem(value); item.setFlags(item.flags() & ~Qt.ItemIsEditable); self.emitter_table.setItem(row,col,item)
        if self.emitters:self.emitter_table.selectRow(self.selected_emitter_index)

    def _emitter_selected(self,row,column):
        if self.emitters and 0<=row<len(self.emitters) and self.emitters[row]["emitter_id"]!=self._details_emitter_id:self._details_emitter_id=None
        super()._emitter_selected(row,column); self._update_mode_history_display()

    def _update_mode_history_display(self):
        self.mode_history_table.clearContents()
        if not self.emitters:return
        e=self.emitters[self.selected_emitter_index]; labels=self.mode_history.labels(e["emitter_id"]); padded=[""]*(10-len(labels))+labels[-10:]
        for col,label in enumerate(padded):
            item=QTableWidgetItem(label); item.setTextAlignment(Qt.AlignCenter); item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if label=="SEARCH":item.setBackground(Qt.lightGray)
            elif label=="DWELL":item.setBackground(Qt.yellow)
            self.mode_history_table.setItem(0,col,item)
        self.mode_history_title.setText(f"{e['emitter_id']} / {e.get('library_id','UNKNOWN')} - RECENT OBSERVED MODES (last 10 s, newest at right)")

    def _show_selected_emitter(self):
        if not self.emitters:
            super()._show_selected_emitter(); self._details_emitter_id=None; return
        e=self.emitters[self.selected_emitter_index]; eid=e["emitter_id"]; scrollbar=self.details.verticalScrollBar(); same=eid==self._details_emitter_id; previous_scroll=scrollbar.value() if same else 0
        super()._show_selected_emitter()
        lib=e.get("library_id","UNKNOWN"); conf=100.*e.get("library_confidence",0.); reason=e.get("library_reason","No library evidence"); existing=self.details.toPlainText()
        # Remove engineering/internal lines from the operator-facing pane.
        remove_prefixes=("Operator assess.","Displayed state","Sequence tracks","ESM threshold","ESM noise floor")
        existing="\n".join(line for line in existing.splitlines() if not line.strip().startswith(remove_prefixes))
        prefix=("EMITTER LIBRARY\n---------------\n"f"ID / confidence  : {lib} / {conf:.0f}%\n"f"Evidence         : {reason}\n\n")
        self.details.setPlainText(prefix+existing)
        scrollbar.setValue(min(previous_scroll,scrollbar.maximum()) if same else 0); self._details_emitter_id=eid


def main():
    app=QApplication(sys.argv); app.setApplicationName("S2B ESM"); window=EnhancedS2BOperatorWindow(); window.show(); sys.exit(app.exec_())


if __name__=="__main__":main()
