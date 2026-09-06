"""Timing profiler for the unified 40 MS/s ESM front end.

Measures wall-clock cost separately for:
  1. simulated IQ source generation,
  2. pulse detection,
  3. PDW extraction including waveform classification.

It also runs an extraction-only baseline that temporarily substitutes a no-op
waveform classifier.  The difference gives a useful estimate of classifier cost
without changing production processing code.
"""

import argparse
import time
from dataclasses import dataclass

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from pdw_extractor import PDWExtractor
from pulse_detector import PulseDetector
from simulated_streaming_source import SimulatedStreamingIQSource


CENTER_HZ = 9_400_000_000
BLOCK_SAMPLES = 40_000


@dataclass
class DummyWaveformResult:
    family: str = "UNKNOWN"
    confidence: float = 0.0
    scores: dict = None
    rejection_reason: str = "timing baseline"

    def __post_init__(self):
        if self.scores is None:
            self.scores = {}


class NoOpWaveformClassifier:
    def classify(self, _pulse_iq):
        return DummyWaveformResult()


def pct(part, total):
    return 0.0 if total <= 0 else 100.0 * part / total


def run_profile(seconds, classify=True):
    source = SimulatedStreamingIQSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_HZ,
        block_samples=BLOCK_SAMPLES,
        noise_std=0.02,
    )
    detector = PulseDetector(DETECTION_THRESHOLD, SAMPLE_RATE_HZ, MIN_PULSE_WIDTH_S)
    extractor = PDWExtractor(SAMPLE_RATE_HZ, CENTER_HZ)
    if not classify:
        extractor.waveform_classifier = NoOpWaveformClassifier()

    blocks = int(round(seconds * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    source_s = 0.0
    detect_s = 0.0
    extract_s = 0.0
    pulses = 0
    pdws = 0

    wall0 = time.perf_counter()
    for _ in range(blocks):
        t0 = time.perf_counter()
        iq, metadata = source.read_block()
        t1 = time.perf_counter()
        found = detector.detect(iq)
        t2 = time.perf_counter()
        for pulse in found:
            extractor.extract(iq, pulse, block_start_time_s=metadata["start_time_s"])
        t3 = time.perf_counter()

        source_s += t1 - t0
        detect_s += t2 - t1
        extract_s += t3 - t2
        pulses += len(found)
        pdws += len(found)

    wall_s = time.perf_counter() - wall0
    return {
        "seconds": seconds,
        "blocks": blocks,
        "samples": blocks * BLOCK_SAMPLES,
        "source_s": source_s,
        "detect_s": detect_s,
        "extract_s": extract_s,
        "wall_s": wall_s,
        "pulses": pulses,
        "pdws": pdws,
    }


def print_run(title, r):
    accounted = r["source_s"] + r["detect_s"] + r["extract_s"]
    other = max(0.0, r["wall_s"] - accounted)
    effective_msps = r["samples"] / r["wall_s"] / 1e6
    realtime_factor = r["wall_s"] / r["seconds"]
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Simulated RF time : {r['seconds']:8.3f} s")
    print(f"Wall-clock time   : {r['wall_s']:8.3f} s")
    print(f"Real-time factor  : {realtime_factor:8.2f} x  (1.00 x = real time)")
    print(f"Effective rate    : {effective_msps:8.2f} MS/s")
    print(f"Blocks            : {r['blocks']:8d}")
    print(f"Samples           : {r['samples']:8d}")
    print(f"Detected pulses   : {r['pulses']:8d}")
    print("\nStage timing")
    print("stage                         seconds    % wall    us/block    us/pulse")
    print("---------------------------  --------  --------  ----------  ----------")
    for name, value in (
        ("IQ simulation/source", r["source_s"]),
        ("Pulse detection", r["detect_s"]),
        ("PDW extract + classifier", r["extract_s"]),
        ("Loop/other", other),
    ):
        us_block = 1e6 * value / max(1, r["blocks"])
        us_pulse = 1e6 * value / max(1, r["pulses"])
        print(f"{name:27s}  {value:8.3f}  {pct(value, r['wall_s']):7.1f}%  {us_block:10.1f}  {us_pulse:10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Profile the unified 40 MS/s streaming ESM front end")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="simulated RF seconds per pass (default: 2.0)")
    args = parser.parse_args()
    seconds = max(0.1, float(args.seconds))

    print("S2B 40 MS/s STREAMING TIMING PROFILER")
    print("======================================")
    print(f"Sample rate       : {SAMPLE_RATE_HZ/1e6:.1f} MS/s")
    print(f"Block size        : {BLOCK_SAMPLES} complex samples ({1e3*BLOCK_SAMPLES/SAMPLE_RATE_HZ:.3f} ms)")
    print(f"RF time per pass  : {seconds:.2f} s")
    print("Production pass   : source -> detector -> PDW extraction -> cyclic classifier")
    print("Baseline pass     : same chain, classifier replaced by a no-op")

    full = run_profile(seconds, classify=True)
    print_run("PASS 1 - FULL PRODUCTION FRONT END", full)

    baseline = run_profile(seconds, classify=False)
    print_run("PASS 2 - CLASSIFIER DISABLED BASELINE", baseline)

    classifier_delta = max(0.0, full["extract_s"] - baseline["extract_s"])
    print("\nDIFFERENTIAL ESTIMATE")
    print("---------------------")
    print(f"PDW extraction with classifier : {full['extract_s']:.3f} s")
    print(f"PDW extraction without it      : {baseline['extract_s']:.3f} s")
    print(f"Estimated classifier increment : {classifier_delta:.3f} s")
    if full["pulses"]:
        print(f"Estimated classifier / pulse   : {1e6*classifier_delta/full['pulses']:.1f} us")

    print("\nInterpretation")
    biggest = max(
        ((full["source_s"], "IQ simulation/source"),
         (full["detect_s"], "pulse detection"),
         (full["extract_s"], "PDW extraction + waveform classification")),
        key=lambda x: x[0],
    )[1]
    print(f"  Largest measured production stage: {biggest}")
    print("  The simulated-source cost will disappear when an Ettus supplies IQ,")
    print("  but SDR transfer/queue cost will replace part of it.")
    print("  Do not infer Ettus real-time capability from source-generation cost alone.")


if __name__ == "__main__":
    main()
