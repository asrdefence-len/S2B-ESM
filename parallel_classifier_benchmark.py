"""Multicore benchmark for the S2B waveform-classification hot path.

This intentionally leaves the working streaming ESM unchanged.  It first builds
one representative set of detected pulse snippets from the existing 40 MS/s
simulator, then classifies exactly the same snippets with 1, 2, 4 and 6 worker
processes.  multiprocessing avoids the Python GIL and lets macOS schedule work
across the M1 Pro CPU cores.

The benchmark reports raw classifier throughput and the CPU fraction required at
the measured pulse arrival rate.  It also verifies that worker count does not
change the classification results.
"""

import argparse
import multiprocessing as mp
import os
import time

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from operational_waveform_classifier_cyclic_fast import FastCyclicOperationalWaveformClassifier
from pulse_detector import PulseDetector
from simulated_streaming_source import SimulatedStreamingIQSource


CENTER_HZ = 9_400_000_000
BLOCK_SAMPLES = 40_000
_WORKER_CLASSIFIER = None


def _worker_init():
    global _WORKER_CLASSIFIER
    _WORKER_CLASSIFIER = FastCyclicOperationalWaveformClassifier(SAMPLE_RATE_HZ)


def _classify_one(item):
    index, pulse_iq = item
    r = _WORKER_CLASSIFIER.classify(pulse_iq)
    return index, r.family, float(r.confidence), tuple(sorted(r.scores.items())), r.rejection_reason


def collect_snippets(seconds):
    source = SimulatedStreamingIQSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_HZ,
        block_samples=BLOCK_SAMPLES,
        noise_std=0.02,
    )
    detector = PulseDetector(DETECTION_THRESHOLD, SAMPLE_RATE_HZ, MIN_PULSE_WIDTH_S)
    blocks = int(round(seconds * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    snippets = []
    wall0 = time.perf_counter()
    for _ in range(blocks):
        iq, _metadata = source.read_block()
        for pulse in detector.detect(iq):
            start = pulse["start_sample"]
            stop = pulse["stop_sample"]
            # Copy deliberately: in the real architecture snippets cross a queue
            # and must not retain a large parent IQ block indefinitely.
            snippets.append(iq[start:stop].copy())
    wall = time.perf_counter() - wall0
    return snippets, wall


def run_serial(items):
    classifier = FastCyclicOperationalWaveformClassifier(SAMPLE_RATE_HZ)
    t0 = time.perf_counter()
    out = []
    for index, pulse_iq in items:
        r = classifier.classify(pulse_iq)
        out.append((index, r.family, float(r.confidence), tuple(sorted(r.scores.items())), r.rejection_reason))
    return out, time.perf_counter() - t0


def run_pool(items, workers, chunksize):
    # spawn matches macOS behaviour and is also portable to the eventual Linux
    # MiniPC.  Pool startup is excluded by warming every worker before timing.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_worker_init) as pool:
        warm = [(i, items[i % len(items)][1]) for i in range(max(workers * 2, 8))]
        pool.map(_classify_one, warm, chunksize=1)
        t0 = time.perf_counter()
        out = pool.map(_classify_one, items, chunksize=chunksize)
        elapsed = time.perf_counter() - t0
    return out, elapsed


def equivalent(reference, candidate, tol=1e-12):
    if len(reference) != len(candidate):
        return False
    for a, b in zip(reference, candidate):
        if a[0] != b[0] or a[1] != b[1] or a[4] != b[4]:
            return False
        if abs(a[2] - b[2]) > tol:
            return False
        if len(a[3]) != len(b[3]):
            return False
        for (ka, va), (kb, vb) in zip(a[3], b[3]):
            if ka != kb or abs(va - vb) > tol:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark multicore S2B waveform classification")
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="RF seconds used to collect representative pulse snippets")
    parser.add_argument("--workers", default="1,2,4,6",
                        help="comma-separated worker-process counts")
    parser.add_argument("--chunksize", type=int, default=16,
                        help="pool task chunk size (default 16)")
    args = parser.parse_args()

    seconds = max(1.0, float(args.seconds))
    workers_requested = []
    for token in args.workers.split(","):
        n = int(token.strip())
        if n > 0 and n not in workers_requested:
            workers_requested.append(n)
    chunksize = max(1, int(args.chunksize))

    print("S2B M1/MULTICORE WAVEFORM CLASSIFIER BENCHMARK")
    print("==============================================")
    print(f"Logical CPUs       : {os.cpu_count()}")
    print(f"Sample rate        : {SAMPLE_RATE_HZ/1e6:.1f} MS/s")
    print(f"RF collection time : {seconds:.2f} s")
    print(f"Workers requested  : {workers_requested}")
    print(f"Pool chunk size    : {chunksize}")
    print()

    snippets, collect_wall = collect_snippets(seconds)
    if not snippets:
        raise RuntimeError("No pulses detected; cannot benchmark classifier")
    items = list(enumerate(snippets))
    pulse_rate = len(items) / seconds
    print(f"Detected snippets  : {len(items)}")
    print(f"Pulse arrival rate : {pulse_rate:.1f} pulses/s")
    print(f"Collection wall    : {collect_wall:.3f} s (not included in classifier timing)")
    print(f"Pulse samples      : min={min(map(len, snippets))} median={sorted(map(len, snippets))[len(snippets)//2]} max={max(map(len, snippets))}")

    reference, serial_s = run_serial(items)
    serial_rate = len(items) / serial_s
    print("\nClassifier throughput")
    print("workers   seconds   pulses/s   us/pulse   speedup   CPU budget at measured pulse rate   exact")
    print("-------  --------  ---------  ---------  --------  -----------------------------  -----")
    budget = pulse_rate / serial_rate
    print(f"serial   {serial_s:8.3f}  {serial_rate:9.0f}  {1e6/serial_rate:9.1f}  {1.0:8.2f}  {100*budget:27.1f}%  REF")

    for workers in workers_requested:
        out, elapsed = run_pool(items, workers, chunksize)
        rate = len(items) / elapsed
        speedup = rate / serial_rate
        budget = pulse_rate / rate
        exact = equivalent(reference, out)
        print(f"{workers:7d}  {elapsed:8.3f}  {rate:9.0f}  {1e6/rate:9.1f}  {speedup:8.2f}  {100*budget:27.1f}%  {'YES' if exact else 'NO'}")

    print("\nInterpretation")
    print("  CPU budget below 100% means that classifier configuration can sustain the")
    print("  measured pulse arrival rate in isolation. This benchmark does not yet include")
    print("  inter-process IQ acquisition, PDW extraction, association, behaviour or UI.")
    print("  The next architecture step is a bounded producer/consumer pipeline once we")
    print("  know whether process-level parallelism actually scales on this machine.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
