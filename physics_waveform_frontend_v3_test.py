from collections import defaultdict

import numpy as np

from physics_waveform_frontend_v2 import PhysicsWaveformFrontendV2
from physics_waveform_frontend_v3 import PhysicsWaveformFrontendV3
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
    v2 = PhysicsWaveformFrontendV2(FS)
    v3 = PhysicsWaveformFrontendV3(FS)
    pulse_widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    trials = 40
    seed = 200000

    stats = {"V2": defaultdict(lambda: [0, 0]), "V3": defaultdict(lambda: [0, 0])}
    v3_snr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    v3_width = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    confusion = defaultdict(lambda: defaultdict(int))
    gain_stats = defaultdict(lambda: defaultdict(list))

    for width in pulse_widths_us:
        num_samples = int(round(width * 1e-6 * FS))
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
                    r2 = v2.classify(observed)
                    r3 = v3.classify(observed)
                    for version, result in (("V2", r2), ("V3", r3)):
                        stats[version][name][0] += int(result.coarse_family == truth)
                        stats[version][name][1] += 1
                    ok = r3.coarse_family == truth
                    v3_snr[name][snr_db][0] += int(ok)
                    v3_snr[name][snr_db][1] += 1
                    v3_width[name][width][0] += int(ok)
                    v3_width[name][width][1] += 1
                    confusion[truth][r3.coarse_family] += 1
                    if name in ("CW", "LFM"):
                        gain_stats[name][snr_db].append(r3.linear_model_gain)

    names = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")
    print("S2B PHYSICS WAVEFORM FRONT END: V2 vs V3")
    print("=========================================")
    print("V3 changes only the smooth CW/LFM model decision; V2 phase-code logic is retained.")
    print()
    print("Overall accuracy by waveform")
    print("----------------------------")
    print("waveform                V2          V3")
    for name in names:
        row = []
        for version in ("V2", "V3"):
            good, total = stats[version][name]
            row.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(row))

    print()
    print("V3 accuracy by SNR")
    print("------------------")
    print("waveform             0dB       5dB      10dB      15dB      20dB")
    for name in names:
        vals = []
        for snr in snrs_db:
            good, total = v3_snr[name][snr]
            vals.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(vals))

    print()
    print("V3 accuracy by pulse width")
    print("--------------------------")
    print("waveform            2.5us      4.5us      7.0us      9.0us")
    for name in names:
        vals = []
        for width in pulse_widths_us:
            good, total = v3_width[name][width]
            vals.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(vals))

    families = ("CW", "LFM", "PHASE_CODED", "UNKNOWN")
    print()
    print("V3 coarse confusion matrix (counts)")
    print("-----------------------------------")
    print("truth\\pred       " + " ".join(f"{family:>12s}" for family in families))
    for truth in ("CW", "LFM", "PHASE_CODED"):
        vals = " ".join(f"{confusion[truth][pred]:12d}" for pred in families)
        print(f"{truth:14s} {vals}")

    print()
    print("Median linear-model gain: CW versus LFM")
    print("---------------------------------------")
    print("SNR             CW          LFM")
    for snr in snrs_db:
        cw_gain = np.median(gain_stats["CW"][snr])
        lfm_gain = np.median(gain_stats["LFM"][snr])
        print(f"{snr:2d} dB   {cw_gain:10.3f}   {lfm_gain:10.3f}")

    for version in ("V2", "V3"):
        good = sum(v[0] for v in stats[version].values())
        total = sum(v[1] for v in stats[version].values())
        print(f"{version} overall accuracy : {100.0 * good / total:.2f}% ({good}/{total})")


if __name__ == "__main__":
    main()
