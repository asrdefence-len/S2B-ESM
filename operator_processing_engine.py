"""Dedicated non-GUI processing process for the live S2B operator display.

The Qt/Matplotlib process should only render low-rate state snapshots. This
module owns the continuous streaming processor, emitter tracker and illumination
behaviour manager in a separate process. The streaming processor in turn owns
its existing four classifier worker processes.
"""

from dataclasses import asdict
import multiprocessing as mp
import queue
import time

from illumination_behaviour import IlluminationBehaviourManager
from parallel_streaming_esm_processor import ParallelStreamingESMProcessor
from simulated_streaming_source import SimulatedStreamingIQSource
from streaming_emitter_tracker import StreamingEmitterTracker


SNAPSHOT_PERIOD_S = 0.100


def _new_chain(config):
    source = SimulatedStreamingIQSource(
        sample_rate_hz=config["sample_rate_hz"],
        center_frequency_hz=config["center_frequency_hz"],
        block_samples=config["block_samples"],
        noise_std=config["noise_std"],
    )
    processor = ParallelStreamingESMProcessor(
        source,
        workers=config["workers"],
        iq_queue_depth=config["iq_queue_depth"],
        classifier_queue_depth=config["classifier_queue_depth"],
        result_queue_depth=config["result_queue_depth"],
        realtime_source=True,
    )
    tracker = StreamingEmitterTracker(frequency_gate_hz=config["frequency_gate_hz"])
    behaviour = IlluminationBehaviourManager(
        illumination_threshold_db=-8.0,
        persistent_s=1.0,
        peak_separation_s=0.25,
        baseline_confidence_threshold=0.50,
        change_confidence_threshold=0.50,
        change_hold_s=5.0,
    )
    return source, processor, tracker, behaviour


def _feed_behaviour(behaviour, track, new_pdws, start_s, end_s, frequency_gate_hz):
    relevant = [p for p in new_pdws if abs(p.frequency_hz-track.frequency_hz) <= frequency_gate_hz]
    bin_s = 0.010
    start = max(0.0, float(start_s)); end = max(start, float(end_s)); by_bin = {}
    for pdw in relevant:
        b = int(pdw.toa_s/bin_s); by_bin[b] = max(by_bin.get(b,-120.0), pdw.amplitude_dbfs)
    assessment = behaviour.assessment(track.emitter_id,start)
    b0=int(start/bin_s); b1=int(__import__("math").ceil(end/bin_s))
    for b in range(b0,b1):
        t=(b+1)*bin_s; assessment=behaviour.update(track.emitter_id,t,by_bin.get(b,-120.0))
    return assessment


def _track_snapshot(track, illumination):
    history=list(track.pri_state_history)
    recent_toas=[float(p.toa_s) for p in list(track.pdws)[-1000:]]
    return {
        "emitter_id":track.emitter_id,
        "frequency_hz":float(track.frequency_hz),
        "receiver_face":int(track.receiver_face),
        "first_seen_s":float(track.first_seen_s),
        "last_seen_s":float(track.last_seen_s),
        "total_pulses":int(track.total_pulses),
        "current":track.summary(),
        "pri_state_history":history,
        "recent_toas":recent_toas,
        "pdw_rate_history":[[int(sec),int(count)] for sec,count in track.pdw_rate_history(12)],
        "illumination":asdict(illumination),
    }


def _publish_latest(snapshot_q,snapshot):
    try:snapshot_q.put_nowait(snapshot); return
    except queue.Full:pass
    try:snapshot_q.get_nowait()
    except queue.Empty:pass
    try:snapshot_q.put_nowait(snapshot)
    except queue.Full:pass


def _engine_main(command_q,snapshot_q,config):
    source,processor,tracker,behaviour=_new_chain(config)
    running=False; last_completed_s=0.0; last_publish_wall=0.0; latest_illumination={}
    try:
        alive=True
        while alive:
            while True:
                try:command=command_q.get_nowait()
                except queue.Empty:break
                if command=="START":
                    if not running:processor.start(); running=True
                elif command=="STOP":
                    if running:processor.stop(); running=False
                elif command=="RESET":
                    if running:processor.stop()
                    source,processor,tracker,behaviour=_new_chain(config); running=False; last_completed_s=0.0; latest_illumination={}
                    _publish_latest(snapshot_q,{"running":False,"status":processor.status(),"tracks":[]})
                elif command=="SHUTDOWN":alive=False; break
            if not alive:break
            if running:
                new_pdws=processor.drain_pdws(); status=processor.status()
                if new_pdws:tracker.update(new_pdws)
                now_s=float(status["completed_time_s"])
                if now_s>last_completed_s:
                    for track in tracker.tracks:
                        latest_illumination[track.emitter_id]=_feed_behaviour(behaviour,track,new_pdws,last_completed_s,now_s,config["frequency_gate_hz"])
                    last_completed_s=now_s
                wall_now=time.perf_counter()
                if wall_now-last_publish_wall>=SNAPSHOT_PERIOD_S:
                    tracks=[]
                    for track in tracker.tracks:
                        illumination=latest_illumination.get(track.emitter_id)
                        if illumination is None:illumination=behaviour.assessment(track.emitter_id,now_s)
                        tracks.append(_track_snapshot(track,illumination))
                    _publish_latest(snapshot_q,{"running":True,"status":status,"tracks":tracks}); last_publish_wall=wall_now
            else:time.sleep(.02)
            if running:time.sleep(.005)
    finally:
        try:processor.stop()
        except Exception:pass


class OperatorProcessingEngine:
    def __init__(self,sample_rate_hz=40_000_000,center_frequency_hz=9_415_000_000,block_samples=40_000,noise_std=.02,workers=4,iq_queue_depth=32,classifier_queue_depth=512,result_queue_depth=512,frequency_gate_hz=2_000_000.0):
        self._ctx=mp.get_context("spawn"); self._command_q=self._ctx.Queue(maxsize=16); self._snapshot_q=self._ctx.Queue(maxsize=3)
        self._config={"sample_rate_hz":float(sample_rate_hz),"center_frequency_hz":float(center_frequency_hz),"block_samples":int(block_samples),"noise_std":float(noise_std),"workers":int(workers),"iq_queue_depth":int(iq_queue_depth),"classifier_queue_depth":int(classifier_queue_depth),"result_queue_depth":int(result_queue_depth),"frequency_gate_hz":float(frequency_gate_hz)}
        self._process=None
    def _ensure_process(self):
        if self._process is not None and self._process.is_alive():return
        self._process=self._ctx.Process(target=_engine_main,args=(self._command_q,self._snapshot_q,self._config),name="S2B-Processing-Engine"); self._process.start()
    def _send(self,command):
        self._ensure_process()
        try:self._command_q.put(command,timeout=.5)
        except queue.Full:raise RuntimeError(f"S2B processing command queue full while sending {command}")
    def start(self):self._send("START")
    def stop(self):
        if self._process is not None and self._process.is_alive():self._send("STOP")
    def reset(self):self._send("RESET"); self._drain_snapshots()
    def latest_snapshot(self):
        latest=None
        while True:
            try:latest=self._snapshot_q.get_nowait()
            except queue.Empty:break
        return latest
    def _drain_snapshots(self):
        while True:
            try:self._snapshot_q.get_nowait()
            except queue.Empty:break
    def shutdown(self,timeout_s=4.0):
        if self._process is None:return
        if self._process.is_alive():
            try:self._command_q.put("SHUTDOWN",timeout=.5)
            except queue.Full:pass
            self._process.join(timeout=float(timeout_s))
            if self._process.is_alive():self._process.terminate(); self._process.join(timeout=1.0)
        self._process=None
