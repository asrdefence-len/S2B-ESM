"""Whole producer/consumer benchmark for the 40 MS/s S2B ESM front end.

This is deliberately a benchmark, not a production-path replacement.  It keeps
40 MS/s IQ generation and pulse detection in the main process, performs the
cheap PDW observables there, and sends only copied pulse snippets plus metadata
to a pool of independent waveform-classifier processes.  That avoids copying
40,000-sample IQ blocks between processes and mirrors the intended architecture:

    SDR/source -> detector/basic PDW -> bounded classifier work -> ordered PDWs

The benchmark compares the existing serial chain with 1/2/4/6 classifier
workers.  Results are returned in input order, so downstream association can
consume a deterministic TOA/PDW stream.  macOS and Linux both use spawn here.
"""

import argparse
import multiprocessing as mp
import os
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
_WORKER_CLASSIFIER = None


@dataclass
class BasicPDW:
    pdw_id: int
    toa_s: float
    pulse_width_s: float
    frequency_hz: float
    amplitude_dbfs: float
    modulation_bandwidth_hz: float


def _worker_init():
    global _WORKER_CLASSIFIER
    _WORKER_CLASSIFIER = FastCyclicOperationalWaveformClassifier(SAMPLE_RATE_HZ)


def _classify_task(task):
    pdw_id, pulse_iq = task
    r = _WORKER_CLASSIFIER.classify(pulse_iq)
    return pdw_id, r.family, float(r.confidence), tuple(sorted(r.scores.items())), r.rejection_reason


def _make_frontend():
    source = SimulatedStreamingIQSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_HZ,
        block_samples=BLOCK_SAMPLES,
        noise_std=0.02,
    )
    detector = PulseDetector(DETECTION_THRESHOLD, SAMPLE_RATE_HZ, MIN_PULSE_WIDTH_S)
    # Reuse the production phase/frequency estimator but do not call extract(),
    # because extract() would also run the classifier in the main process.
    estimator = PDWExtractor(SAMPLE_RATE_HZ, CENTER_HZ)
    return source, detector, estimator


def _basic_pdw(estimator, iq, pulse, block_start_s, pdw_id):
    start = pulse["start_sample"]
    stop = pulse["stop_sample"]
    pulse_iq = iq[start:stop]
    pw_s = (stop - start) / SAMPLE_RATE_HZ
    amp = np.sqrt(np.mean(np.abs(pulse_iq) ** 2))
    amp_dbfs = 20.0 * np.log10(max(float(amp), 1e-12))
    foff, bw = estimator._estimate_frequency_and_bandwidth(pulse_iq, pw_s)
    basic = BasicPDW(
        pdw_id=pdw_id,
        toa_s=float(block_start_s) + start / SAMPLE_RATE_HZ,
        pulse_width_s=pw_s,
        frequency_hz=CENTER_HZ + foff,
        amplitude_dbfs=amp_dbfs,
        modulation_bandwidth_hz=bw,
    )
    return basic, pulse_iq.copy()


def run_serial(seconds):
    source, detector, extractor = _make_frontend()
    blocks = int(round(seconds * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    source_s = detect_s = extract_s = 0.0
    pdws = []
    signatures = []
    wall0 = time.perf_counter()
    for _ in range(blocks):
        t0 = time.perf_counter()
        iq, metadata = source.read_block()
        t1 = time.perf_counter()
        found = detector.detect(iq)
        t2 = time.perf_counter()
        for pulse in found:
            p = extractor.extract(iq, pulse, block_start_time_s=metadata["start_time_s"])
            pdws.append(p)
            signatures.append((p.pdw_id, p.modulation_type, float(p.modulation_confidence),
                               tuple(sorted(p.modulation_scores.items())), p.modulation_rejection_reason))
        t3 = time.perf_counter()
        source_s += t1 - t0
        detect_s += t2 - t1
        extract_s += t3 - t2
    wall_s = time.perf_counter() - wall0
    return {"wall_s": wall_s, "source_s": source_s, "detect_s": detect_s,
            "extract_s": extract_s, "pdws": pdws, "signatures": signatures,
            "blocks": blocks}


def run_parallel(seconds, workers, chunksize):
    source, detector, estimator = _make_frontend()
    blocks = int(round(seconds * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    source_s = detect_s = basic_s = 0.0
    basics = []
    tasks = []
    next_id = 1

    # Front-end pass.  This measures source + detector + basic PDW work and
    # prepares the bounded-size pulse work items.  Classification is then timed
    # independently through the process pool.  The final wall estimate below is
    # the conservative pipeline service bound max(front-end, classifier), plus
    # measured pool IPC/compute time already contained in classifier_s.
    front0 = time.perf_counter()
    for _ in range(blocks):
        t0 = time.perf_counter()
        iq, metadata = source.read_block()
        t1 = time.perf_counter()
        found = detector.detect(iq)
        t2 = time.perf_counter()
        for pulse in found:
            b0 = time.perf_counter()
            basic, snippet = _basic_pdw(estimator, iq, pulse, metadata["start_time_s"], next_id)
            basic_s += time.perf_counter() - b0
            basics.append(basic)
            tasks.append((next_id, snippet))
            next_id += 1
        source_s += t1 - t0
        detect_s += t2 - t1
    frontend_wall = time.perf_counter() - front0

    ctx = mp.get_context("spawn")
    pool0 = time.perf_counter()
    with ctx.Pool(processes=workers, initializer=_worker_init) as pool:
        # Warm workers before the measured classification pass.
        warm = tasks[:max(workers * 2, 8)]
        if warm:
            pool.map(_classify_task, warm, chunksize=1)
        class0 = time.perf_counter()
        signatures = pool.map(_classify_task, tasks, chunksize=chunksize)
        classifier_s = time.perf_counter() - class0
    pool_total_s = time.perf_counter() - pool0

    # If producer and classifier run concurrently with a bounded queue, steady
    # state throughput is governed by the slower side.  Startup/drain is small
    # for long streams but is not hidden here: both measured components are
    # printed separately.  We call this an estimated pipelined wall, not a
    # measured concurrent wall, to avoid overstating the result.
    estimated_pipeline_wall = max(frontend_wall, classifier_s)
    return {"frontend_wall": frontend_wall, "source_s": source_s,
            "detect_s": detect_s, "basic_s": basic_s,
            "classifier_s": classifier_s, "pool_total_s": pool_total_s,
            "estimated_pipeline_wall": estimated_pipeline_wall,
            "pdws": basics, "signatures": signatures, "blocks": blocks}


def signatures_equivalent(reference, candidate, tol=1e-12):
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel 40 MS/s S2B front-end pipeline")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--workers", default="1,2,4,6")
    parser.add_argument("--chunksize", type=int, default=16)
    args = parser.parse_args()
    seconds = max(1.0, float(args.seconds))
    workers_list = [int(x.strip()) for x in args.workers.split(",") if int(x.strip()) > 0]
    chunksize = max(1, int(args.chunksize))

    print("S2B 40 MS/s WHOLE PIPELINE MULTICORE BENCHMARK")
    print("==============================================")
    print(f"Logical CPUs       : {os.cpu_count()}")
    print(f"Sample rate        : {SAMPLE_RATE_HZ/1e6:.1f} MS/s")
    print(f"Block size         : {BLOCK_SAMPLES} ({1e3*BLOCK_SAMPLES/SAMPLE_RATE_HZ:.3f} ms RF)")
    print(f"RF duration        : {seconds:.2f} s")
    print(f"Classifier workers : {workers_list}")
    print(f"Pool chunk size    : {chunksize}")

    serial = run_serial(seconds)
    pulse_rate = len(serial["pdws"]) / seconds
    print("\nSERIAL PRODUCTION REFERENCE")
    print(f"Wall               : {serial['wall_s']:.3f} s")
    print(f"Real-time factor   : {serial['wall_s']/seconds:.3f} x")
    print(f"Effective rate     : {SAMPLE_RATE_HZ*seconds/serial['wall_s']/1e6:.2f} MS/s")
    print(f"PDWs               : {len(serial['pdws'])} ({pulse_rate:.1f}/s)")
    print(f"Source             : {serial['source_s']:.3f} s")
    print(f"Detector           : {serial['detect_s']:.3f} s")
    print(f"PDW + classifier   : {serial['extract_s']:.3f} s")

    print("\nPARALLEL SERVICE-RATE TEST")
    print("workers  frontend  classifier  est.pipe  RTF est  MS/s est  exact")
    print("-------  --------  ----------  --------  -------  --------  -----")
    for workers in workers_list:
        r = run_parallel(seconds, workers, chunksize)
        exact = signatures_equivalent(serial["signatures"], r["signatures"])
        pipe = r["estimated_pipeline_wall"]
        msps = SAMPLE_RATE_HZ * seconds / pipe / 1e6
        print(f"{workers:7d}  {r['frontend_wall']:8.3f}  {r['classifier_s']:10.3f}  {pipe:8.3f}  {pipe/seconds:7.3f}  {msps:8.2f}  {'YES' if exact else 'NO'}")
        print(f"         source={r['source_s']:.3f}s detector={r['detect_s']:.3f}s basicPDW={r['basic_s']:.3f}s poolTotal={r['pool_total_s']:.3f}s")

    print("\nIMPORTANT")
    print("  'est.pipe' is a service-rate estimate: max(front-end wall, classifier wall).")
    print("  It is NOT yet a measured concurrent producer/consumer wall time.")
    print("  If this estimate is comfortably below 1.0x real time, the next test will")
    print("  connect the stages with bounded live queues and measure queue high-water,")
    print("  backpressure/drops and true end-to-end wall time.  No Ettus performance is")
    print("  inferred from the simulator's source-generation cost.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
