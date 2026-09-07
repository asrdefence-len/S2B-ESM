"""Live parallel 40 MS/s processor for the S2B operator display.

The Qt UI must never perform RF processing itself.  This engine owns a continuous
producer/consumer pipeline and exposes only completed, ordered PDWs plus status.

Architecture:
    IQSource -> bounded in-process IQ queue -> detector/basic PDW thread
             -> bounded multiprocessing task queue -> classifier workers
             -> result collector -> ordered completed-PDW deque -> UI

Only short pulse snippets cross process boundaries.  The 40,000-sample IQ blocks
remain in one process.  multiprocessing uses 'spawn' explicitly so the same code
works on macOS/M1 and Linux/Ryzen.
"""

from collections import deque
import multiprocessing as mp
import queue
import threading
import time

import numpy as np

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S
from operational_waveform_classifier_cyclic_fast import FastCyclicOperationalWaveformClassifier
from pdw import PDW
from pdw_extractor import PDWExtractor
from pulse_detector import PulseDetector


_STOP = None


def _classifier_worker(task_q, result_q, sample_rate_hz):
    classifier = FastCyclicOperationalWaveformClassifier(sample_rate_hz)
    while True:
        item = task_q.get()
        if item is _STOP:
            return
        pdw_id, pulse_iq = item
        r = classifier.classify(pulse_iq)
        result_q.put((
            pdw_id,
            r.family,
            float(r.confidence),
            dict(r.scores),
            r.rejection_reason,
        ))


class ParallelStreamingESMProcessor:
    """Background streaming engine that supplies ordered PDWs to a GUI consumer."""

    def __init__(
        self,
        source,
        workers=4,
        iq_queue_depth=32,
        classifier_queue_depth=512,
        result_queue_depth=512,
        realtime_source=True,
    ):
        self.source = source
        self.workers = int(workers)
        self.iq_queue_depth = int(iq_queue_depth)
        self.classifier_queue_depth = int(classifier_queue_depth)
        self.result_queue_depth = int(result_queue_depth)
        self.realtime_source = bool(realtime_source)

        self.detector = PulseDetector(
            DETECTION_THRESHOLD, source.sample_rate_hz, MIN_PULSE_WIDTH_S
        )
        # Reuse the production RF/PW/amplitude estimator, but classification is
        # deliberately done only in worker processes.
        self.estimator = PDWExtractor(source.sample_rate_hz, source.center_frequency_hz)

        self._ctx = mp.get_context("spawn")
        self._iq_q = queue.Queue(maxsize=self.iq_queue_depth)
        self._task_q = None
        self._result_q = None
        self._processes = []
        self._threads = []
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._completed = deque()
        self._basic = {}
        self._pending_results = {}

        self.running = False
        self.total_blocks = 0
        self.total_pdws = 0
        self.completed_pdws = 0
        self.stream_time_s = 0.0
        self.iq_high_water = 0
        self.producer_wait_s = 0.0
        self.classifier_wait_s = 0.0
        self.ordering_errors = 0
        self._next_pdw_id = 1
        self._next_output_id = 1

    def _new_process_queues(self):
        self._task_q = self._ctx.Queue(maxsize=self.classifier_queue_depth)
        self._result_q = self._ctx.Queue(maxsize=self.result_queue_depth)

    def start(self):
        if self.running:
            return
        self._stop_event.clear()
        self._new_process_queues()
        self._processes = [
            self._ctx.Process(
                target=_classifier_worker,
                args=(self._task_q, self._result_q, self.source.sample_rate_hz),
                daemon=True,
            )
            for _ in range(self.workers)
        ]
        for p in self._processes:
            p.start()

        self._threads = [
            threading.Thread(target=self._producer_loop, name="s2b-iq-producer", daemon=True),
            threading.Thread(target=self._frontend_loop, name="s2b-pdw-frontend", daemon=True),
            threading.Thread(target=self._collector_loop, name="s2b-pdw-collector", daemon=True),
        ]
        self.running = True
        for t in self._threads:
            t.start()

    def stop(self, timeout_s=3.0):
        if not self.running:
            return
        self._stop_event.set()
        # Wake the in-process front-end if it is waiting for IQ.
        try:
            self._iq_q.put_nowait(_STOP)
        except queue.Full:
            pass

        deadline = time.perf_counter() + float(timeout_s)
        for t in self._threads:
            remaining = max(0.0, deadline - time.perf_counter())
            t.join(timeout=remaining)

        for _ in self._processes:
            try:
                self._task_q.put_nowait(_STOP)
            except Exception:
                break
        for p in self._processes:
            remaining = max(0.0, deadline - time.perf_counter())
            p.join(timeout=remaining)
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)

        self.running = False
        self._threads = []
        self._processes = []
        if self._task_q is not None:
            self._task_q.close()
        if self._result_q is not None:
            self._result_q.close()

    def reset(self):
        self.stop()
        self.source.reset()
        self._iq_q = queue.Queue(maxsize=self.iq_queue_depth)
        with self._output_lock:
            self._completed.clear()
        with self._state_lock:
            self._basic.clear()
            self._pending_results.clear()
            self.total_blocks = 0
            self.total_pdws = 0
            self.completed_pdws = 0
            self.stream_time_s = 0.0
            self.iq_high_water = 0
            self.producer_wait_s = 0.0
            self.classifier_wait_s = 0.0
            self.ordering_errors = 0
            self._next_pdw_id = 1
            self._next_output_id = 1

    def drain_pdws(self, max_count=None):
        out = []
        with self._output_lock:
            count = len(self._completed) if max_count is None else min(len(self._completed), int(max_count))
            for _ in range(count):
                out.append(self._completed.popleft())
        return out

    def status(self):
        with self._state_lock:
            submitted = self.total_pdws
            completed = self.completed_pdws
            return {
                "running": self.running,
                "stream_time_s": self.stream_time_s,
                "total_blocks": self.total_blocks,
                "total_pdws": submitted,
                "completed_pdws": completed,
                "classifier_backlog": max(0, submitted - completed),
                "iq_queue": self._iq_q.qsize(),
                "iq_queue_depth": self.iq_queue_depth,
                "iq_high_water": self.iq_high_water,
                "classifier_queue_depth": self.classifier_queue_depth,
                "producer_wait_s": self.producer_wait_s,
                "classifier_wait_s": self.classifier_wait_s,
                "ordering_errors": self.ordering_errors,
                "workers": self.workers,
                "drops": 0,
            }

    def _put_with_stop(self, q, item, wait_counter=None):
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                q.put(item, timeout=0.05)
                waited = time.perf_counter() - t0
                if wait_counter is not None:
                    with self._state_lock:
                        setattr(self, wait_counter, getattr(self, wait_counter) + waited)
                return True
            except queue.Full:
                if wait_counter is not None:
                    with self._state_lock:
                        setattr(self, wait_counter, getattr(self, wait_counter) + (time.perf_counter() - t0))
        return False

    def _producer_loop(self):
        wall_start = time.perf_counter()
        rf_start = self.source.time_s
        block_duration_s = self.source.block_samples / self.source.sample_rate_hz
        while not self._stop_event.is_set():
            iq, metadata = self.source.read_block()
            if not self._put_with_stop(self._iq_q, (iq, metadata), "producer_wait_s"):
                break
            with self._state_lock:
                self.iq_high_water = max(self.iq_high_water, self._iq_q.qsize())

            # A real SDR naturally delivers samples at the RF sample clock.  Pace
            # the simulator the same way so scenario time equals operator wall time.
            if self.realtime_source:
                rf_elapsed = (metadata["start_time_s"] + block_duration_s) - rf_start
                target_wall = wall_start + rf_elapsed
                delay = target_wall - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

    def _basic_pdw(self, iq, pulse, block_start_s, pdw_id):
        start = pulse["start_sample"]
        stop = pulse["stop_sample"]
        snippet = iq[start:stop].copy()
        pw_s = (stop - start) / self.source.sample_rate_hz
        amplitude_linear = np.sqrt(np.mean(np.abs(snippet) ** 2))
        amplitude_dbfs = 20.0 * np.log10(max(float(amplitude_linear), 1e-12))
        frequency_offset_hz, bandwidth_hz = self.estimator._estimate_frequency_and_bandwidth(snippet, pw_s)
        return snippet, {
            "pdw_id": pdw_id,
            "toa_s": float(block_start_s) + start / self.source.sample_rate_hz,
            "pulse_width_s": pw_s,
            "frequency_hz": self.source.center_frequency_hz + frequency_offset_hz,
            "amplitude_dbfs": amplitude_dbfs,
            "aoa_deg": self.estimator.default_aoa_deg,
            "modulation_bandwidth_hz": bandwidth_hz,
        }

    def _frontend_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._iq_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is _STOP:
                break
            iq, metadata = item
            pulses = self.detector.detect(iq)
            for pulse in pulses:
                with self._state_lock:
                    pdw_id = self._next_pdw_id
                    self._next_pdw_id += 1
                snippet, basic = self._basic_pdw(iq, pulse, metadata["start_time_s"], pdw_id)
                with self._state_lock:
                    self._basic[pdw_id] = basic
                    self.total_pdws += 1
                if not self._put_with_stop(self._task_q, (pdw_id, snippet), "classifier_wait_s"):
                    return

            with self._state_lock:
                self.total_blocks += 1
                self.stream_time_s = metadata["start_time_s"] + self.source.block_samples / self.source.sample_rate_hz

    def _collector_loop(self):
        while not self._stop_event.is_set() or self.completed_pdws < self.total_pdws:
            try:
                result = self._result_q.get(timeout=0.05)
            except queue.Empty:
                continue
            pdw_id, family, confidence, scores, reason = result
            with self._state_lock:
                self._pending_results[pdw_id] = (family, confidence, scores, reason)
                while self._next_output_id in self._pending_results and self._next_output_id in self._basic:
                    oid = self._next_output_id
                    family, confidence, scores, reason = self._pending_results.pop(oid)
                    basic = self._basic.pop(oid)
                    pdw = PDW(
                        modulation_type=family,
                        modulation_confidence=confidence,
                        modulation_scores=scores,
                        modulation_rejection_reason=reason,
                        **basic,
                    )
                    with self._output_lock:
                        self._completed.append(pdw)
                    self.completed_pdws += 1
                    if oid != self._next_output_id:
                        self.ordering_errors += 1
                    self._next_output_id += 1
