"""Strict V7 -> V8 regression and timing test on measured 40 MS/s pulse snippets."""

import argparse
import time

import numpy as np

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from physics_waveform_frontend_v7 import PhysicsWaveformFrontendV7
from physics_waveform_frontend_v8 import PhysicsWaveformFrontendV8
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
    for x in snippets:
        frontend.classify(x)  # also warms V8 caches for production steady-state timing
    t0 = time.perf_counter()
    for _ in range(repeats):
        for x in snippets:
            frontend.classify(x)
    return time.perf_counter() - t0, repeats * len(snippets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pulses", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    snippets = collect_pulses(max(20, args.pulses))
    repeats = max(1, args.repeats)

    print("S2B PHYSICS WAVEFORM FRONTEND V7 -> V8 REGRESSION")
    print("=================================================")
    lengths = np.asarray([len(x) for x in snippets])
    print(f"Measured pulse snippets: {len(snippets)}")
    print(f"Samples/pulse          : min={lengths.min()} median={np.median(lengths):.0f} max={lengths.max()}")
    print(f"Unique pulse lengths   : {len(set(lengths.tolist()))}")

    v7 = PhysicsWaveformFrontendV7(SAMPLE_RATE_HZ)
    v8 = PhysicsWaveformFrontendV8(SAMPLE_RATE_HZ)
    matches = 0
    max_rms_error = 0.0
    max_score_error = 0.0
    max_excursion_error = 0.0
    max_plateau_error = 0.0
    mismatches = []

    for i, x in enumerate(snippets):
        a = v7.classify(x)
        b = v8.classify(x)
        if a.family == b.family:
            matches += 1
        elif len(mismatches) < 10:
            mismatches.append((i, a.family, b.family))
        for av, bv in ((a.phase_rms_linear, b.phase_rms_linear),
                       (a.phase_rms_quadratic, b.phase_rms_quadratic),
                       (a.phase_rms_cubic, b.phase_rms_cubic)):
            max_rms_error = max(max_rms_error, abs(av - bv))
        for key in set(a.scores) | set(b.scores):
            max_score_error = max(max_score_error, abs(a.scores.get(key, 0.0) - b.scores.get(key, 0.0)))
        max_excursion_error = max(max_excursion_error, abs(a.local_frequency_excursion_hz - b.local_frequency_excursion_hz))
        max_plateau_error = max(max_plateau_error, abs(a.local_frequency_plateau_fraction - b.local_frequency_plateau_fraction))

    t7, calls = timed(v7, snippets, repeats)
    t8, _ = timed(v8, snippets, repeats)
    us7 = 1e6 * t7 / calls
    us8 = 1e6 * t8 / calls

    print("\nEvidence regression")
    print("-------------------")
    print(f"Family agreement              : {matches}/{len(snippets)} ({100*matches/len(snippets):.2f}%)")
    print(f"Max phase-fit RMS error       : {max_rms_error:.12e}")
    print(f"Max classification-score error: {max_score_error:.12e}")
    print(f"Max frequency-excursion error : {max_excursion_error:.12e} Hz")
    print(f"Max plateau-fraction error    : {max_plateau_error:.12e}")
    if mismatches:
        for row in mismatches:
            print(f"  pulse {row[0]}: V7={row[1]} V8={row[2]}")

    print("\nTiming - warmed steady state")
    print("----------------------------")
    print(f"V7 full physics frontend : {us7:9.1f} us/pulse")
    print(f"V8 full physics frontend : {us8:9.1f} us/pulse")
    print(f"Frontend speed-up        : {t7/t8 if t8 else float('inf'):9.2f} x")
    print(f"V8 cached projections    : {len(v8._phase_projection_cache)}")

    equivalent = (
        matches == len(snippets)
        and max_rms_error < 1e-10
        and max_score_error < 1e-10
        and max_excursion_error < 1e-6
        and max_plateau_error < 1e-12
    )
    print("\nDecision")
    if equivalent and t8 < t7:
        print("PASS: V8 preserves V7 evidence and is faster after cache warm-up.")
        print("Next: promote V8 to the operational classifier and rerun streaming timing.")
    else:
        print("HOLD: do not promote V8 yet.")
        if not equivalent:
            print("  Evidence differs beyond strict regression tolerances.")
        if t8 >= t7:
            print("  V8 was not faster in this run.")


if __name__ == "__main__":
    main()
