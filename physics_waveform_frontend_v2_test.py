from collections import defaultdict

import numpy as np

from physics_waveform_frontend import PhysicsWaveformFrontend
from physics_waveform_frontend_v2 import PhysicsWaveformFrontendV2
from waveform_classifier_stress_test import impair
from waveform_library import barker13, biphase, cw, lfm, polyphase


FS = 40_000_000.0


def clean_cases(num_samples):
    return (
        ("CW", "CW", cw(num_samples)),
        ("LFM", "LFM", lfm(num_samples, 0.25)),
        ("BARKER13", "PHASE_CODED", barker13(num_samples)),
        ("GENERIC_BIPHASE", "PHASE_CODED", biphase(num_samples)),
        ("FRANK16_LIKE", "PHASE_CODED", polyphase(num_samples)),
    )


def main():
    v1 = PhysicsWaveformFrontend(FS)
    v2 = PhysicsWaveformFrontendV2(FS)
    pulse_widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    trials = 40
    seed = 200000  # same impairment seed sequence as the V1 test

    by_version = {
        "V1": defaultdict(lambda: [0, 0]),
        "V2": defaultdict(lambda: [0, 0]),
    }
    v2_by_snr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    v2_by_width = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    confusion = defaultdict(lambda: defaultdict(int))
    families = ("CW", "LFM", "PHASE_CODED", "UNKNOWN")

    for pulse_width_us in pulse_widths_us:
        num_samples = int(round(pulse_width_us * 1e-6 * FS))
        for name, truth, clean in clean_cases(num_samples):
            for snr_db in snrs_db:
                for _ in range(trials):
                    seed += 1
                    rng = np.random.default_rng(seed)
                    observed = impair(
                        clean,
                        snr_db=snr_db,
                        frequency_offset_hz=rng.uniform(-5e6, 5e6),
                        phase_rad=rng.uniform(-np.pi, np.pi),
                        timing_shift=int(rng.integers(-3, 4)),
                        seed=seed,
                    )
                    r1 = v1.classify(observed)
                    r2 = v2.classify(observed)
                    ok1 = r1.coarse_family == truth
                    ok2 = r2.coarse_family == truth
                    by_version["V1"][name][0] += int(ok1)
                    by_version["V1"][name][1] += 1
                    by_version["V2"][name][0] += int(ok2)
                    by_version["V2"][name][1] += 1
                    v2_by_snr[name][snr_db][0] += int(ok2)
                    v2_by_snr[name][snr_db][1] += 1
                    v2_by_width[name][pulse_width_us][0] += int(ok2)
                    v2_by_width[name][pulse_width_us][1] += 1
                    confusion[truth][r2.coarse_family] += 1

    names = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")

    print("S2B PHYSICS WAVEFORM FRONT END: V1 vs V2")
    print("=========================================")
    print("Same 4000 impaired pulses are classified by both versions.")
    print()
    print("Overall accuracy by waveform")
    print("----------------------------")
    print("waveform                V1          V2")
    for name in names:
        vals = []
        for version in ("V1", "V2"):
            good, total = by_version[version][name]
            vals.append(f"{100.0 * good / total:9.2f}%")
        print(f"{name:18s}" + " ".join(vals))

    print()
    print("V2 accuracy by SNR")
    print("------------------")
    print("waveform             0dB       5dB      10dB      15dB      20dB")
    for name in names:
        vals = []
        for snr in snrs_db:
            good, total = v2_by_snr[name][snr]
            vals.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(vals))

    print()
    print("V2 accuracy by pulse width")
    print("--------------------------")
    print("waveform            2.5us      4.5us      7.0us      9.0us")
    for name in names:
        vals = []
        for width in pulse_widths_us:
            good, total = v2_by_width[name][width]
            vals.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(vals))

    print()
    print("V2 coarse confusion matrix (counts)")
    print("-----------------------------------")
    print("truth\\pred       " + " ".join(f"{family:>12s}" for family in families))
    for truth in ("CW", "LFM", "PHASE_CODED"):
        values = " ".join(f"{confusion[truth][pred]:12d}" for pred in families)
        print(f"{truth:14s} {values}")

    for version in ("V1", "V2"):
        good = sum(value[0] for value in by_version[version].values())
        total = sum(value[1] for value in by_version[version].values())
        print(f"{version} overall accuracy : {100.0 * good / total:.2f}% ({good}/{total})")


if __name__ == "__main__":
    main()
