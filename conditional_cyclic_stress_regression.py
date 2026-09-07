"""Broad waveform stress regression for conditional cyclic classification.

Compares the current full cyclic classifier against the experimental conditional
classifier across clean and impaired CW/FM/phase-coded/multi-component waveforms.
Promotion is allowed only if family decisions, rejection reasons, scores and
confidence remain equivalent over the stress set while phase-coded cases still
exercise x^2/x^4 cyclic processing.
"""

import argparse
import time
from collections import defaultdict

import numpy as np

from cyclostationary_waveform_diagnostic import FS, make_waveform
from operational_waveform_classifier_cyclic import CyclicOperationalWaveformClassifier
from operational_waveform_classifier_cyclic_fast import FastCyclicOperationalWaveformClassifier
from waveform_classifier_stress_test import impair


WAVEFORMS = (
    "CW",
    "LFM",
    "NLFM",
    "STEPPED_FREQ",
    "BARKER13",
    "GENERIC_BIPHASE",
    "FRANK16_LIKE",
    "RANDOM_POLYPHASE",
    "DUAL_TONE",
    "HYBRID_FM_PHASE",
)


def main():
    parser = argparse.ArgumentParser(description="Stress-regress conditional cyclic classifier")
    parser.add_argument("--trials", type=int, default=30, help="trials per waveform/SNR")
    parser.add_argument("--width-us", type=float, default=7.0)
    args = parser.parse_args()

    trials = max(5, int(args.trials))
    count = int(round(float(args.width_us) * 1e-6 * FS))
    snrs = (20, 15, 10, 5)
    full = CyclicOperationalWaveformClassifier(FS)
    fast = FastCyclicOperationalWaveformClassifier(FS)

    total = 0
    family_matches = 0
    reason_matches = 0
    max_score_error = 0.0
    max_conf_error = 0.0
    x24_count = 0
    phase_x24_count = 0
    rows = defaultdict(lambda: {"n": 0, "family": 0, "reason": 0, "x24": 0})
    mismatches = []
    samples = []
    seed = 920000

    print("S2B CONDITIONAL CYCLIC BROAD WAVEFORM STRESS REGRESSION")
    print("========================================================")
    print(f"Waveforms         : {len(WAVEFORMS)}")
    print(f"SNRs              : {snrs}")
    print(f"Trials/case       : {trials}")
    print(f"Samples/waveform  : {count}")
    print()

    for snr in snrs:
        for name in WAVEFORMS:
            key = (snr, name)
            for _ in range(trials):
                seed += 1
                rng = np.random.default_rng(seed)
                clean = make_waveform(name, count, rng)
                obs = impair(
                    clean,
                    snr_db=snr,
                    frequency_offset_hz=rng.uniform(-5e6, 5e6),
                    phase_rad=rng.uniform(-np.pi, np.pi),
                    timing_shift=int(rng.integers(-3, 4)),
                    seed=seed,
                )
                a = full.classify(obs)
                b = fast.classify(obs)
                samples.append(obs)
                total += 1
                rows[key]["n"] += 1

                fam_ok = a.family == b.family
                reason_ok = a.rejection_reason == b.rejection_reason
                if fam_ok:
                    family_matches += 1
                    rows[key]["family"] += 1
                if reason_ok:
                    reason_matches += 1
                    rows[key]["reason"] += 1

                used = bool(b.cyclic_features.get("x2_x4_evaluated", False))
                if used:
                    x24_count += 1
                    rows[key]["x24"] += 1
                    if name in ("BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE", "RANDOM_POLYPHASE", "HYBRID_FM_PHASE"):
                        phase_x24_count += 1

                keys = set(a.scores) | set(b.scores)
                score_error = max(abs(a.scores.get(k, 0.0) - b.scores.get(k, 0.0)) for k in keys)
                conf_error = abs(a.confidence - b.confidence)
                max_score_error = max(max_score_error, score_error)
                max_conf_error = max(max_conf_error, conf_error)

                if (not fam_ok or not reason_ok or score_error > 1e-12 or conf_error > 1e-12) and len(mismatches) < 20:
                    mismatches.append((snr, name, a.family, b.family, a.rejection_reason, b.rejection_reason, score_error, conf_error, used))

    print("Per-case agreement")
    print("SNR  waveform              family   reason   x2/x4")
    print("---  --------------------  -------  -------  -----")
    for snr in snrs:
        for name in WAVEFORMS:
            r = rows[(snr, name)]
            print(f"{snr:3d}  {name:20s}  {r['family']:2d}/{r['n']:<2d}   {r['reason']:2d}/{r['n']:<2d}   {r['x24']:2d}/{r['n']:<2d}")

    print("\nOverall regression")
    print("------------------")
    print(f"Family agreement        : {family_matches}/{total} ({100*family_matches/total:.2f}%)")
    print(f"Rejection-reason match  : {reason_matches}/{total} ({100*reason_matches/total:.2f}%)")
    print(f"Max score error         : {max_score_error:.12e}")
    print(f"Max confidence error    : {max_conf_error:.12e}")
    print(f"x2/x4 FFTs evaluated    : {x24_count}/{total} ({100*x24_count/total:.2f}%)")
    print(f"Phase-like x2/x4 uses   : {phase_x24_count}")

    if mismatches:
        print("\nFirst mismatches")
        print("----------------")
        for row in mismatches:
            print(
                f"SNR={row[0]:2d} {row[1]:18s} full={row[2]:12s} fast={row[3]:12s} "
                f"score_err={row[6]:.3e} conf_err={row[7]:.3e} x2/x4={row[8]}\n"
                f"    full reason={row[4]!r}\n"
                f"    fast reason={row[5]!r}"
            )

    # Timing on exactly the same stress observations.
    for x in samples[:20]:
        full.classify(x)
        fast.classify(x)
    t0 = time.perf_counter()
    for x in samples:
        full.classify(x)
    full_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for x in samples:
        fast.classify(x)
    fast_s = time.perf_counter() - t0

    print("\nStress-set timing")
    print("-----------------")
    print(f"Full cyclic classifier : {1e6*full_s/total:9.1f} us/pulse")
    print(f"Conditional classifier : {1e6*fast_s/total:9.1f} us/pulse")
    print(f"Speed-up               : {full_s/fast_s if fast_s else float('inf'):9.2f} x")

    exact = (
        family_matches == total
        and reason_matches == total
        and max_score_error < 1e-12
        and max_conf_error < 1e-12
    )
    exercised = x24_count > 0 and phase_x24_count > 0

    print("\nDecision")
    if exact and exercised and fast_s < full_s:
        print("PASS: conditional classifier is decision-equivalent across the broad stress set,")
        print("      phase-like cases still exercise x2/x4 processing, and the stress set is faster.")
        print("Next: promote the conditional classifier and rerun the full 40 MS/s streaming profiler.")
    else:
        print("HOLD: do not promote the conditional classifier yet.")
        if not exact:
            print("  Full and conditional classification differ on the stress set.")
        if not exercised:
            print("  The test did not exercise x2/x4 on phase-like cases; conditional gating needs review.")
        if fast_s >= full_s:
            print("  The conditional classifier was not faster on this stress set.")


if __name__ == "__main__":
    main()
