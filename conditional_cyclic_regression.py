"""Compare full and conditional cyclic classifiers on measured streaming pulses."""

import argparse
import time

import numpy as np

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from operational_waveform_classifier_cyclic import CyclicOperationalWaveformClassifier
from operational_waveform_classifier_cyclic_fast import FastCyclicOperationalWaveformClassifier
from pulse_detector import PulseDetector
from simulated_streaming_source import SimulatedStreamingIQSource

CENTER_HZ = 9_400_000_000
BLOCK_SAMPLES = 40_000


def collect_pulses(target):
    source = SimulatedStreamingIQSource(sample_rate_hz=SAMPLE_RATE_HZ,
                                        center_frequency_hz=CENTER_HZ,
                                        block_samples=BLOCK_SAMPLES,
                                        noise_std=0.02)
    detector = PulseDetector(DETECTION_THRESHOLD, SAMPLE_RATE_HZ, MIN_PULSE_WIDTH_S)
    out = []
    while len(out) < target:
        iq, _ = source.read_block()
        for pulse in detector.detect(iq):
            x = iq[pulse["start_sample"]:pulse["stop_sample"]]
            if len(x) >= 24:
                out.append(x.copy())
                if len(out) >= target:
                    break
    return out


def timed(classifier, snippets, repeats):
    for x in snippets[:20]:
        classifier.classify(x)
    t0 = time.perf_counter()
    for _ in range(repeats):
        for x in snippets:
            classifier.classify(x)
    elapsed = time.perf_counter() - t0
    return elapsed, repeats * len(snippets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pulses", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    snippets = collect_pulses(max(50, args.pulses))
    repeats = max(1, args.repeats)

    reference = CyclicOperationalWaveformClassifier(SAMPLE_RATE_HZ)
    fast = FastCyclicOperationalWaveformClassifier(SAMPLE_RATE_HZ)
    family_matches = 0
    reason_matches = 0
    max_score_error = 0.0
    max_conf_error = 0.0
    x24_evaluated = 0
    mismatches = []

    for i, x in enumerate(snippets):
        a = reference.classify(x)
        b = fast.classify(x)
        family_matches += int(a.family == b.family)
        reason_matches += int(a.rejection_reason == b.rejection_reason)
        max_conf_error = max(max_conf_error, abs(a.confidence - b.confidence))
        for key in set(a.scores) | set(b.scores):
            max_score_error = max(max_score_error, abs(a.scores.get(key, 0.0) - b.scores.get(key, 0.0)))
        x24_evaluated += int(b.cyclic_features.get("x2_x4_evaluated", False))
        if a.family != b.family and len(mismatches) < 10:
            mismatches.append((i, a.family, b.family))

    tref, calls = timed(reference, snippets, repeats)
    tfast, _ = timed(fast, snippets, repeats)
    ref_us = 1e6 * tref / calls
    fast_us = 1e6 * tfast / calls

    print("S2B CONDITIONAL CYCLIC CLASSIFIER REGRESSION")
    print("============================================")
    print(f"Measured pulse snippets : {len(snippets)}")
    print(f"Samples/pulse           : min={min(map(len,snippets))} median={np.median(list(map(len,snippets))):.0f} max={max(map(len,snippets))}")
    print("\nDecision regression")
    print("-------------------")
    print(f"Family agreement        : {family_matches}/{len(snippets)} ({100*family_matches/len(snippets):.2f}%)")
    print(f"Rejection-reason match  : {reason_matches}/{len(snippets)} ({100*reason_matches/len(snippets):.2f}%)")
    print(f"Max score error         : {max_score_error:.12e}")
    print(f"Max confidence error    : {max_conf_error:.12e}")
    print(f"x2/x4 FFTs evaluated    : {x24_evaluated}/{len(snippets)} ({100*x24_evaluated/len(snippets):.2f}%)")
    if mismatches:
        for row in mismatches:
            print(f"  pulse {row[0]}: reference={row[1]} conditional={row[2]}")

    print("\nTiming")
    print("------")
    print(f"Full cyclic classifier  : {ref_us:9.1f} us/pulse")
    print(f"Conditional classifier  : {fast_us:9.1f} us/pulse")
    print(f"Speed-up                : {tref/tfast if tfast else float('inf'):9.2f} x")

    equivalent = (family_matches == len(snippets) and reason_matches == len(snippets)
                  and max_score_error < 1e-12 and max_conf_error < 1e-12)
    print("\nDecision")
    if equivalent and tfast < tref:
        print("PASS: conditional cyclic processing preserves decisions and is faster on this measured stream.")
        print("Do not promote yet: next run the broad waveform stress suite to protect difficult phase-coded/UNKNOWN cases.")
    else:
        print("HOLD: do not promote the conditional classifier.")


if __name__ == "__main__":
    main()
