"""S2B ESM main operator display using the validated live parallel pipeline.

This preserves the enhanced operator picture and replaces only its serial RF
processing plumbing. Qt consumes completed PDWs; it never performs IQ/pulse/
waveform processing on the GUI thread.
"""

import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QPushButton
from esm_operator_ui_enhanced import EnhancedS2BOperatorWindow
from emitter_measurement_scatter import EmitterMeasurementScatterWindow
from illumination_behaviour import IlluminationBehaviourManager
from parallel_streaming_esm_processor import ParallelStreamingESMProcessor
from simulated_streaming_source import SimulatedStreamingIQSource
from streaming_emitter_tracker import StreamingEmitterTracker


class LiveS2BOperatorWindow(EnhancedS2BOperatorWindow):
    WORKERS=4; IQ_QUEUE_DEPTH=32; CLASSIFIER_QUEUE_DEPTH=512; RESULT_QUEUE_DEPTH=512
    PLOT_REFRESH_MS=1000

    def __init__(self):
        super().__init__(); self.setWindowTitle("S2B ESM - Live Parallel 40 MS/s Operator Display"); self.timer.setInterval(100)
        self.measurement_plot=None
        self.measurement_plot_timer=QTimer(self)
        self.measurement_plot_timer.setInterval(self.PLOT_REFRESH_MS)
        self.measurement_plot_timer.timeout.connect(self._update_measurement_plot)
        root=self.centralWidget().layout(); top=root.itemAt(0).layout() if root is not None and root.count() else None
        self.measurement_plot_button=QPushButton("PRI / SIGNAL PLOT")
        self.measurement_plot_button.clicked.connect(self._open_measurement_plot)
        if top is not None: top.addWidget(self.measurement_plot_button)

    def _shutdown_stream(self):
        processor=getattr(self,"stream_processor",None)
        if processor is not None and hasattr(processor,"stop"): processor.stop()

    def _new_stream(self):
        self._shutdown_stream()
        # One 40 MS/s sector only. Centre at 9.415 GHz so E3 at 9.410 GHz and the
        # two simple test emitters near 9.42 GHz all sit comfortably in-band.
        self.stream_source=SimulatedStreamingIQSource(sample_rate_hz=40_000_000,center_frequency_hz=9_415_000_000,block_samples=self.BLOCK_SAMPLES,noise_std=.02)
        self.stream_processor=ParallelStreamingESMProcessor(self.stream_source,workers=self.WORKERS,iq_queue_depth=self.IQ_QUEUE_DEPTH,classifier_queue_depth=self.CLASSIFIER_QUEUE_DEPTH,result_queue_depth=self.RESULT_QUEUE_DEPTH,realtime_source=True)
        self.stream_tracker=StreamingEmitterTracker(frequency_gate_hz=2_000_000.0)
        self.behaviour=IlluminationBehaviourManager(illumination_threshold_db=-8.0,persistent_s=1.0,peak_separation_s=.25,baseline_confidence_threshold=.50,change_confidence_threshold=.50,change_hold_s=5.0)
        self.last_stream_time_s=0.0

    def _selected_track(self):
        if not self.emitters or self.stream_tracker is None:return None
        eid=self.emitters[self.selected_emitter_index]["emitter_id"]
        return next((t for t in self.stream_tracker.tracks if t.emitter_id==eid),None)

    def _open_measurement_plot(self):
        if self.measurement_plot is None:self.measurement_plot=EmitterMeasurementScatterWindow(self,max_points=2000)
        self.measurement_plot.show();self.measurement_plot.raise_();self.measurement_plot.activateWindow();self._update_measurement_plot()
        if not self.measurement_plot_timer.isActive():self.measurement_plot_timer.start()

    def _update_measurement_plot(self):
        if self.measurement_plot is None:return
        if not self.measurement_plot.isVisible():self.measurement_plot_timer.stop();return
        track=self._selected_track();lib=None
        if self.emitters:lib=self.emitters[self.selected_emitter_index].get("library_id")
        self.measurement_plot.update_track(track,lib)

    def start_system(self):
        if self.running:return
        if self.stream_processor is None:self._new_stream()
        self.stream_processor.start();self.running=True;self._set_status("RUNNING");self.timer.start();self._refresh()
        if self.measurement_plot is not None and self.measurement_plot.isVisible() and not self.measurement_plot_timer.isActive():self.measurement_plot_timer.start()

    def stop_system(self):
        if not self.running:return
        self.running=False;self.timer.stop();self.measurement_plot_timer.stop();self._shutdown_stream();self._set_status("STOPPED")
        status=self.stream_processor.status() if self.stream_processor is not None else {}
        self.statusBar().showMessage(f"Stopped | PDWs {status.get('completed_pdws',0)} | IQ high-water {status.get('iq_high_water',0)}/{status.get('iq_queue_depth',self.IQ_QUEUE_DEPTH)} | drops {status.get('drops',0)}")

    def reset_system(self):
        self.timer.stop();self.measurement_plot_timer.stop();self.running=False;self._shutdown_stream();self.mode_history.clear();self.library_memory.clear();self.library_degraded.clear();self.library_operator_confirmed.clear();self.watched_emitters.clear();self.operator_assessments.clear();self._details_emitter_id=None;self._new_stream();self._show_prestart_blank();self._set_status("STOPPED");self._update_measurement_plot()

    def _refresh(self):
        if not self.running:return
        try:
            new_pdws=self.stream_processor.drain_pdws();status=self.stream_processor.status()
            if new_pdws:self.stream_tracker.update(new_pdws)
            now_s=status["completed_time_s"]
            if now_s<=self.last_stream_time_s:
                self.statusBar().showMessage(f"STREAM 40.0 MS/s | acquiring/classifying | RF {status['stream_time_s']:6.2f} s | safe {now_s:6.2f} s | lag {1000*status['pipeline_lag_s']:.0f} ms | PDWs {status['completed_pdws']} | IQ Q {status['iq_queue']}/{status['iq_queue_depth']} | CLASS backlog {status['classifier_backlog']}/{status['classifier_queue_depth']} | DROPS {status['drops']}");return
            out=[]
            for track in self.stream_tracker.tracks:
                current=track.summary();illum=self._feed_behaviour(track,new_pdws,now_s);system_state=illum.system_assessment if illum.state!="UNRESOLVED" else ("MONITOR" if track.total_pulses>=3 else "UNASSESSED");eid=track.emitter_id
                e={"emitter_id":eid,"aoa_deg":self._display_aoa(track),"system_state":system_state,"state":system_state,"display_color":self._assessment_color(system_state),"watched":eid in self.watched_emitters,"tracks":[],"current":current,"links":[],"track_confidence":min(1.0,track.total_pulses/20.0),"illumination":illum,"last_seen_s":track.last_seen_s}
                self._apply_operator_assessment(e);self._assign_library(e);out.append(e)
                if illum.state in ("PERIODIC_SCAN","PERSISTENT_ILLUMINATION"):self.mode_history.update(eid,max(0.,now_s-.010),illum.state)
            self.emitters=out
            if self.selected_emitter_index>=len(out):self.selected_emitter_index=max(0,len(out)-1)
            self._populate_table();self.polar.update_emitters(out,self.selected_emitter_index);self._show_selected_emitter();self._update_mode_history_display();self.last_stream_time_s=now_s
            self.statusBar().showMessage(f"STREAM 40.0 MS/s | RF {status['stream_time_s']:6.2f} s | safe {now_s:6.2f} s | lag {1000*status['pipeline_lag_s']:.0f} ms | PDWs {status['completed_pdws']} | IQ Q {status['iq_queue']}/{status['iq_queue_depth']} (max {status['iq_high_water']}) | CLASS backlog {status['classifier_backlog']}/{status['classifier_queue_depth']} | WORKERS {status['workers']} | DROPS {status['drops']}")
        except Exception as exc:
            self.timer.stop();self.measurement_plot_timer.stop();self.running=False;self._shutdown_stream();self._set_status("ERROR");self.details.setPlainText(f"Live streaming UI error:\n\n{exc}");self.statusBar().showMessage(str(exc))

    def _emitter_selected(self,row,column):
        super()._emitter_selected(row,column);self._update_measurement_plot()

    @staticmethod
    def _assessment_color(state):
        from esm_operator_ui import ASSESSMENT_COLORS
        return ASSESSMENT_COLORS.get(state,ASSESSMENT_COLORS["UNASSESSED"])

    def closeEvent(self,event):
        self.timer.stop();self.measurement_plot_timer.stop();self.running=False;self._shutdown_stream()
        if self.measurement_plot is not None:self.measurement_plot.close()
        event.accept()


def main():
    import multiprocessing as mp
    mp.freeze_support();app=QApplication(sys.argv);app.setApplicationName("S2B ESM");window=LiveS2BOperatorWindow();window.show();sys.exit(app.exec_())


if __name__=="__main__":main()
