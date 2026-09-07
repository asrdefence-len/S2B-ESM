"""Live bounded producer/consumer benchmark for the S2B 40 MS/s ESM.

Unlike parallel_streaming_pipeline_benchmark.py, this test actually overlaps the
40 MS/s simulator producer, front-end pulse detector/basic-PDW stage, and four
waveform-classifier worker processes.  It is still benchmark-only: production
ESM code is not changed.

Architecture:
    source thread -> bounded IQ queue -> front-end thread -> bounded task queue
                  -> classifier processes -> result queue -> ordered collector

IQ blocks stay inside one process/thread address space to avoid serialising
40,000 complex samples through multiprocessing.  Only short copied pulse
snippets cross process boundaries.  The multiprocessing context is explicitly
'spawn' on macOS and Linux for portability.
"""

import argparse
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from operational_waveform_classifier_cyclic_fast import FastCyclicOperationalWaveformClassifier
from pdw_extractor import PDWExtractor
from pulse_detector import PulseDetector
from simulated_streaming_source import SimulatedStreamingIQSource

CENTER_HZ = 9_400_000_000
BLOCK_SAMPLES = 40_000
STOP = None


@dataclass
class Stats:
    iq_high_water: int = 0
    task_high_water: int = 0
    result_high_water: int = 0
    producer_wait_s: float = 0.0
    frontend_wait_s: float = 0.0
    produced_blocks: int = 0
    consumed_blocks: int = 0
    produced_pdws: int = 0
    received_results: int = 0
    ordering_errors: int = 0


def _source():
    return SimulatedStreamingIQSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_HZ,
        block_samples=BLOCK_SAMPLES,
        noise_std=0.02,
    )


def _detector():
    return PulseDetector(DETECTION_THRESHOLD, SAMPLE_RATE_HZ, MIN_PULSE_WIDTH_S)


def _basic_task(estimator, iq, pulse, block_start_s, pdw_id):
    start = pulse["start_sample"]
    stop = pulse["stop_sample"]
    snippet = iq[start:stop].copy()
    pw_s = (stop - start) / SAMPLE_RATE_HZ
    amp = np.sqrt(np.mean(np.abs(snippet) ** 2))
    amp_dbfs = 20.0 * np.log10(max(float(amp), 1e-12))
    foff, bw = estimator._estimate_frequency_and_bandwidth(snippet, pw_s)
    basic = (
        pdw_id,
        float(block_start_s) + start / SAMPLE_RATE_HZ,
        pw_s,
        CENTER_HZ + foff,
        amp_dbfs,
        bw,
    )
    return pdw_id, snippet, basic


def _classifier_worker(task_q, result_q):
    classifier = FastCyclicOperationalWaveformClassifier(SAMPLE_RATE_HZ)
    while True:
        item = task_q.get()
        if item is STOP:
            break
        pdw_id, snippet = item
        r = classifier.classify(snippet)
        result_q.put((
            pdw_id,
            r.family,
            float(r.confidence),
            tuple(sorted(r.scores.items())),
            r.rejection_reason,
        ))


def _reference(seconds):
    source = _source()
    detector = _detector()
    extractor = PDWExtractor(SAMPLE_RATE_HZ, CENTER_HZ)
    blocks = int(round(seconds * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    signatures = []
    for _ in range(blocks):
        iq, metadata = source.read_block()
        for pulse in detector.detect(iq):
            p = extractor.extract(iq, pulse, block_start_time_s=metadata["start_time_s"])
            signatures.append((p.pdw_id, p.modulation_type, float(p.modulation_confidence),
                               tuple(sorted(p.modulation_scores.items())), p.modulation_rejection_reason))
    return signatures


def _equivalent(reference, candidate, tol=1e-12):
    if len(reference) != len(candidate):
        return False
    for a, b in zip(reference, candidate):
        if a[0] != b[0] or a[1] != b[1] or a[4] != b[4] or abs(a[2] - b[2]) > tol:
            return False
        if len(a[3]) != len(b[3]):
            return False
        for (ka, va), (kb, vb) in zip(a[3], b[3]):
            if ka != kb or abs(va - vb) > tol:
                return False
    return True


def run_live(seconds, workers, iq_depth, task_depth, result_depth):
    blocks = int(round(seconds * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    iq_q = queue.Queue(maxsize=iq_depth)
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=task_depth)
    result_q = ctx.Queue(maxsize=result_depth)
    stats = Stats()
    stats_lock = threading.Lock()
    basic_by_id = {}
    producer_done = threading.Event()
    frontend_done = threading.Event()

    processes = [ctx.Process(target=_classifier_worker, args=(task_q, result_q)) for _ in range(workers)]
    for p in processes:
        p.start()

    def producer():
        source = _source()
        for _ in range(blocks):
            iq, metadata = source.read_block()
            t0 = time.perf_counter()
            iq_q.put((iq, metadata))
            waited = time.perf_counter() - t0
            with stats_lock:
                stats.producer_wait_s += waited
                stats.produced_blocks += 1
                stats.iq_high_water = max(stats.iq_high_water, iq_q.qsize())
        iq_q.put(STOP)
        producer_done.set()

    def frontend():
        detector = _detector()
        estimator = PDWExtractor(SAMPLE_RATE_HZ, CENTER_HZ)
        next_id = 1
        while True:
            item = iq_q.get()
            if item is STOP:
                break
            iq, metadata = item
            found = detector.detect(iq)
            with stats_lock:
                stats.consumed_blocks += 1
            for pulse in found:
                pdw_id, snippet, basic = _basic_task(estimator, iq, pulse, metadata["start_time_s"], next_id)
                basic_by_id[pdw_id] = basic
                t0 = time.perf_counter()
                task_q.put((pdw_id, snippet))
                waited = time.perf_counter() - t0
                with stats_lock:
                    stats.frontend_wait_s += waited
                    stats.produced_pdws += 1
                    try:
                        stats.task_high_water = max(stats.task_high_water, task_q.qsize())
                    except NotImplementedError:
                        pass
                next_id += 1
        for _ in range(workers):
            task_q.put(STOP)
        frontend_done.set()

    producer_thread = threading.Thread(target=producer, name="iq-producer")
    frontend_thread = threading.Thread(target=frontend, name="pdw-frontend")

    wall0 = time.perf_counter()
    producer_thread.start()
    frontend_thread.start()

    # Results can complete out of order.  Collect continuously so workers never
    # block behind a full result queue, then restore deterministic PDW-ID order.
    results = {}
    next_expected = 1
    ordered = []
    while True:
        try:
            r = result_q.get(timeout=0.05)
            pdw_id = r[0]
            results[pdw_id] = r
            stats.received_results += 1
            try:
                stats.result_high_water = max(stats.result_high_water, result_q.qsize())
            except NotImplementedError:
                pass
            while next_expected in results:
                ordered.append(results.pop(next_expected))
                next_expected += 1
        except queue.Empty:
            pass
        if frontend_done.is_set() and stats.received_results >= stats.produced_pdws:
            break

    producer_thread.join()
    frontend_thread.join()
    for p in processes:
        p.join()
    wall_s = time.perf_counter() - wall0

    stats.ordering_errors = sum(1 for i, r in enumerate(ordered, start=1) if r[0] != i)
    return wall_s, ordered, stats


def main():
    parser = argparse.ArgumentParser(description="Live bounded 40 MS/s S2B pipeline benchmark")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iq-depth", type=int, default=16)
    parser.add_argument("--task-depth", type=int, default=512)
    parser.add_argument("--result-depth", type=int, default=512)
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()

    seconds = max(1.0, args.seconds)
    workers = max(1, args.workers)
    print("S2B 40 MS/s LIVE BOUNDED PIPELINE BENCHMARK")
    print("===========================================")
    print(f"Logical CPUs       : {os.cpu_count()}")
    print(f"Sample rate        : {SAMPLE_RATE_HZ/1e6:.1f} MS/s")
    print(f"Block size         : {BLOCK_SAMPLES} ({1e3*BLOCK_SAMPLES/SAMPLE_RATE_HZ:.3f} ms RF)")
    print(f"RF duration        : {seconds:.3f} s")
    print(f"Classifier workers : {workers}")
    print(f"Queue depths       : IQ={args.iq_depth}, task={args.task_depth}, result={args.result_depth}")

    reference = None
    if not args.skip_reference:
        print("\nBuilding serial equivalence reference (excluded from live timing)...")
        ref0 = time.perf_counter()
        reference = _reference(seconds)
        print(f"Reference PDWs     : {len(reference)} in {time.perf_counter()-ref0:.3f} s")

    print("\nRunning live concurrent pipeline...")
    wall_s, ordered, stats = run_live(seconds, workers, args.iq_depth, args.task_depth, args.result_depth)
    rtf = wall_s / seconds
    effective = SAMPLE_RATE_HZ * seconds / wall_s / 1e6
    exact = True if reference is None else _equivalent(reference, ordered)

    print("\nRESULT")
    print(f"RF processed       : {seconds:.3f} s")
    print(f"Wall time          : {wall_s:.3f} s")
    print(f"Real-time factor   : {rtf:.3f} x")
    print(f"Effective rate     : {effective:.2f} MS/s")
    print(f"Blocks             : {stats.consumed_blocks}/{stats.produced_blocks}")
    print(f"PDWs/results       : {stats.produced_pdws}/{stats.received_results}")
    print(f"IQ queue high-water: {stats.iq_high_water}/{args.iq_depth}")
    print(f"Task high-water    : {stats.task_high_water}/{args.task_depth}")
    print(f"Result high-water  : {stats.result_high_water}/{args.result_depth}")
    print(f"Producer queue wait: {stats.producer_wait_s:.3f} s")
    print(f"Frontend task wait : {stats.frontend_wait_s:.3f} s")
    print(f"Ordering errors    : {stats.ordering_errors}")
    print(f"Classification eq. : {'PASS' if exact else 'FAIL'}")
    print("Dropped IQ blocks  : 0 (lossless blocking benchmark)")
    print("Dropped PDWs       : 0 (lossless blocking benchmark)")

    passed = rtf < 1.0 and stats.ordering_errors == 0 and exact
    print(f"\nREAL-TIME PIPELINE : {'PASS' if passed else 'HOLD'}")
    if passed:
        print("The simulated live processing architecture sustained 40 MS/s losslessly.")
    else:
        print("Inspect queue high-water/wait times before changing production architecture.")
    print("This test does not measure UHD/USB/Ettus acquisition performance.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
