"""Profile the expensive pieces of the operational cyclic waveform classifier.

Uses real detected pulse snippets from the unified 40 MS/s stream, then times
classifier internals repeatedly without changing production classifier code.
"""

import argparse
import time

import numpy as np

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from cyclostationary_waveform_diagnostic import cyclic_features
from operational_waveform_classifier_cyclic import CyclicOperationalWaveformClassifier
from physics_waveform_frontend_v6 import PhysicsWaveformFrontendV6
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
    snippets = []
    while len(snippets) < target:
        iq, _ = source.read_block()
        for pulse in detector.detect(iq):
            x = iq[pulse["start_sample"]:pulse["stop_sample"]]
            if len(x) >= 24:
                snippets.append(x.copy())
                if len(snippets) >= target:
                    break
    return snippets


def timed(fn, snippets, repeats):
    # Warm NumPy dispatch/caches before measuring.
    for x in snippets[:min(10, len(snippets))]:
        fn(x)
    t0 = time.perf_counter()
    calls = 0
    for _ in range(repeats):
        for x in snippets:
            fn(x)
            calls += 1
    elapsed = time.perf_counter() - t0
    return elapsed, calls


def main():
    parser = argparse.ArgumentParser(description="Profile waveform-classifier internals")
    parser.add_argument("--pulses", type=int, default=300,
                        help="number of real detected pulse snippets to profile")
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeat passes over snippets")
    args = parser.parse_args()
    count = max(20, args.pulses)
    repeats = max(1, args.repeats)

    print("S2B WAVEFORM CLASSIFIER INTERNAL TIMING PROFILER")
    print("================================================")
    print("Collecting pulse snippets from the unified 40 MS/s detector...")
    snippets = collect_pulses(count)
    lengths = np.asarray([len(x) for x in snippets])
    print(f"Pulse snippets     : {len(snippets)}")
    print(f"Samples/pulse      : min={lengths.min()} median={np.median(lengths):.0f} max={lengths.max()}")
    print(f"Repeated passes    : {repeats}\n")

    frontend = PhysicsWaveformFrontendV6(SAMPLE_RATE_HZ)
    full = CyclicOperationalWaveformClassifier(SAMPLE_RATE_HZ)

    stages = []
    for name, fn in (
        ("Physics frontend V6", frontend.classify),
        ("Cyclic features x2/x4/env", cyclic_features),
        ("Full cyclic operational", full.classify),
    ):
        elapsed, calls = timed(fn, snippets, repeats)
        stages.append((name, elapsed, calls))

    print("Direct component timing")
    print("component                       seconds   us/call    calls/s")
    print("------------------------------  --------  ---------  ---------")
    for name, elapsed, calls in stages:
        print(f"{name:30s}  {elapsed:8.3f}  {1e6*elapsed/calls:9.1f}  {calls/elapsed:9.0f}")

    front_us = 1e6 * stages[0][1] / stages[0][2]
    cyclic_us = 1e6 * stages[1][1] / stages[1][2]
    full_us = 1e6 * stages[2][1] / stages[2][2]
    overhead_us = max(0.0, full_us - front_us - cyclic_us)

    print("\nApproximate decomposition per pulse")
    print(f"  Physics frontend       : {front_us:9.1f} us")
    print(f"  Cyclic features        : {cyclic_us:9.1f} us")
    print(f"  Wrapper/scoring        : {overhead_us:9.1f} us")
    print(f"  Full classifier        : {full_us:9.1f} us")

    # Profile the known suspicious local-frequency routine separately. It performs
    # a least-squares solve at every position in a pulse.
    local_elapsed, local_calls = timed(
        lambda x: frontend._local_frequency(np.unwrap(np.angle(np.asarray(x, dtype=np.complex128)))),
        snippets,
        repeats,
    )
    local_us = 1e6 * local_elapsed / local_calls

    fit1_elapsed, fit1_calls = timed(
        lambda x: frontend._phase_fit_rms(np.unwrap(np.angle(np.asarray(x, dtype=np.complex128))), 1),
        snippets,
        repeats,
    )
    fit2_elapsed, fit2_calls = timed(
        lambda x: frontend._phase_fit_rms(np.unwrap(np.angle(np.asarray(x, dtype=np.complex128))), 2),
        snippets,
        repeats,
    )
    fit3_elapsed, fit3_calls = timed(
        lambda x: frontend._phase_fit_rms(np.unwrap(np.angle(np.asarray(x, dtype=np.complex128))), 3),
        snippets,
        repeats,
    )
    jump_elapsed, jump_calls = timed(frontend._wrapped_phase_jump_score, snippets, repeats)

    print("\nPhysics-front-end hotspots")
    print("operation                        us/call")
    print("------------------------------  ---------")
    print(f"local frequency sliding LSQ     {local_us:9.1f}")
    print(f"linear phase polyfit             {1e6*fit1_elapsed/fit1_calls:9.1f}")
    print(f"quadratic phase polyfit          {1e6*fit2_elapsed/fit2_calls:9.1f}")
    print(f"cubic phase polyfit              {1e6*fit3_elapsed/fit3_calls:9.1f}")
    print(f"wrapped phase-jump score         {1e6*jump_elapsed/jump_calls:9.1f}")

    print("\nLikely interpretation")
    if local_us > 0.5 * front_us:
        print("  The sliding local-frequency least-squares loop is the dominant physics cost.")
        print("  It is an excellent optimization target: estimate local frequency with vectorized")
        print("  phase differences / convolution rather than np.linalg.lstsq at every sample.")
    elif cyclic_us > front_us:
        print("  The three cyclic FFT feature sets dominate; optimize or conditionally invoke them.")
    else:
        print("  Cost is distributed; use the measured table before choosing an optimization.")
    print("  This profiler changes no production classifier behaviour.")


if __name__ == "__main__":
    main()
