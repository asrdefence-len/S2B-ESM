"""Regression and timing comparison for PhysicsWaveformFrontendV6 vs V7.

V7 should preserve V6 classification/evidence while replacing the expensive
sliding np.linalg.lstsq loop with vectorized closed-form linear regression.
"""

import argparse
import time

import numpy as np

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from physics_waveform_frontend_v6 import PhysicsWaveformFrontendV6
from physics_waveform_frontend_v7 import PhysicsWaveformFrontendV7
from pulse_detector import PulseDetector
from simulated_streaming_source import SimulatedStreamingIQSource


CENTER_HZ = 9_400_000_000
BLOCK_SAMPLES = 40_000


def collect_pulses(target):
    source = SimulatedStreamingIQSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_HZ,
        block_samples=BLOCK_SAMPLES,
        noise_std=0.02,
    )
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


def timed(frontend, snippets, repeats):
    for x in snippets[:10]:
        frontend.classify(x)
    t0 = time.perf_counter()
    results = []
    for _ in range(repeats):
        results = [frontend.classify(x) for x in snippets]
    elapsed = time.perf_counter() - t0
    return elapsed, results, repeats * len(snippets)


def main():
    parser = argparse.ArgumentParser(description="Compare V6 and vectorized V7 waveform frontends")
    parser.add_argument("--pulses", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    count = max(20, args.pulses)
    repeats = max(1, args.repeats)

    print("S2B PHYSICS WAVEFORM FRONTEND V6 -> V7 REGRESSION")
    print("=================================================")
    snippets = collect_pulses(count)
    lengths = np.asarray([len(x) for x in snippets])
    print(f"Measured pulse snippets: {len(snippets)}")
    print(f"Samples/pulse          : min={lengths.min()} median={np.median(lengths):.0f} max={lengths.max()}")

    v6 = PhysicsWaveformFrontendV6(SAMPLE_RATE_HZ)
    v7 = PhysicsWaveformFrontendV7(SAMPLE_RATE_HZ)

    # One-pass paired evidence comparison.
    family_matches = 0
    max_excursion_error = 0.0
    max_plateau_error = 0.0
    max_score_error = 0.0
    mismatches = []
    for i, x in enumerate(snippets):
        a = v6.classify(x)
        b = v7.classify(x)
        if a.family == b.family:
            family_matches += 1
        elif len(mismatches) < 10:
            mismatches.append((i, a.family, b.family))
        max_excursion_error = max(max_excursion_error, abs(a.local_frequency_excursion_hz - b.local_frequency_excursion_hz))
        max_plateau_error = max(max_plateau_error, abs(a.local_frequency_plateau_fraction - b.local_frequency_plateau_fraction))
        for key in set(a.scores) | set(b.scores):
            max_score_error = max(max_score_error, abs(a.scores.get(key, 0.0) - b.scores.get(key, 0.0)))

    v6_s, _, calls = timed(v6, snippets, repeats)
    v7_s, _, _ = timed(v7, snippets, repeats)
    v6_us = 1e6 * v6_s / calls
    v7_us = 1e6 * v7_s / calls
    speedup = v6_s / v7_s if v7_s > 0 else float("inf")

    # Isolate local-frequency implementation timing on identical unwrapped phases.
    phases = [np.unwrap(np.angle(np.asarray(x, dtype=np.complex128))) for x in snippets]
    t0 = time.perf_counter()
    for _ in range(repeats):
        for p in phases:
            v6._local_frequency(p)
    local6_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(repeats):
        for p in phases:
            v7._local_frequency(p)
    local7_s = time.perf_counter() - t0

    print("\nEvidence regression")
    print("-------------------")
    print(f"Family agreement              : {family_matches}/{len(snippets)} ({100*family_matches/len(snippets):.2f}%)")
    print(f"Max local-frequency excursion : {max_excursion_error:.6f} Hz difference")
    print(f"Max plateau-fraction error    : {max_plateau_error:.12f}")
    print(f"Max classification-score error: {max_score_error:.12f}")
    if mismatches:
        print("First family mismatches:")
        for row in mismatches:
            print(f"  pulse {row[0]}: V6={row[1]} V7={row[2]}")

    print("\nTiming")
    print("------")
    print(f"V6 full physics frontend : {v6_us:9.1f} us/pulse")
    print(f"V7 full physics frontend : {v7_us:9.1f} us/pulse")
    print(f"Full frontend speed-up   : {speedup:9.2f} x")
    print(f"V6 local frequency       : {1e6*local6_s/calls:9.1f} us/pulse")
    print(f"V7 local frequency       : {1e6*local7_s/calls:9.1f} us/pulse")
    print(f"Local-frequency speed-up : {local6_s/local7_s if local7_s > 0 else float('inf'):9.2f} x")

    exact_enough = (
        family_matches == len(snippets)
        and max_plateau_error < 1e-12
        and max_score_error < 1e-10
        and max_excursion_error < 1e-3
    )
    faster = v7_s < v6_s
    print("\nDecision")
    if exact_enough and faster:
        print("PASS: V7 preserves V6 decisions/evidence and is faster on measured pulse snippets.")
        print("Next: place V7 behind the cyclic operational classifier and rerun the full classifier regression/timing.")
    else:
        print("HOLD: do not promote V7 yet.")
        if not exact_enough:
            print("  Evidence differs beyond the strict numerical regression tolerance.")
        if not faster:
            print("  V7 was not faster in this run.")


if __name__ == "__main__":
    main()
