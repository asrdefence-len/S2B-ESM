from collections import defaultdict

import numpy as np

from physics_waveform_frontend import PhysicsWaveformFrontend
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
    classifier = PhysicsWaveformFrontend(FS)
    pulse_widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    trials = 40
    seed = 200000

    by_waveform_snr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_waveform_width = defaultdict(lambda: defaultdict(lambda: [0, 0]))
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
                    result = classifier.classify(observed)
                    ok = result.coarse_family == truth
                    by_waveform_snr[name][snr_db][0] += int(ok)
                    by_waveform_snr[name][snr_db][1] += 1
                    by_waveform_width[name][pulse_width_us][0] += int(ok)
                    by_waveform_width[name][pulse_width_us][1] += 1
                    confusion[truth][result.coarse_family] += 1

    names = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")

    print("S2B PHYSICS-INFORMED WAVEFORM FRONT-END TEST")
    print("============================================")
    print("Coarse classes : CW / LFM / PHASE_CODED / UNKNOWN")
    print("Purpose        : route easy physics first; PCA later resolves coded family")
    print()
    print("Accuracy by SNR")
    print("---------------")
    print("waveform             0dB       5dB      10dB      15dB      20dB")
    for name in names:
        values = []
        for snr_db in snrs_db:
            good, total = by_waveform_snr[name][snr_db]
            values.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(values))

    print()
    print("Accuracy by pulse width")
    print("-----------------------")
    print("waveform            2.5us      4.5us      7.0us      9.0us")
    for name in names:
        values = []
        for width in pulse_widths_us:
            good, total = by_waveform_width[name][width]
            values.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(values))

    print()
    print("Coarse confusion matrix (counts)")
    print("--------------------------------")
    print("truth\\pred       " + " ".join(f"{family:>12s}" for family in families))
    for truth in ("CW", "LFM", "PHASE_CODED"):
        values = " ".join(f"{confusion[truth][pred]:12d}" for pred in families)
        print(f"{truth:14s} {values}")

    total_good = 0
    total_count = 0
    for by_snr in by_waveform_snr.values():
        for good, count in by_snr.values():
            total_good += good
            total_count += count
    print()
    print(f"Overall coarse accuracy : {100.0 * total_good / total_count:.2f}% ({total_good}/{total_count})")
    print()
    print("This is intentionally only the first stage. PHASE_CODED pulses would next")
    print("go to the FFT/PCA evidence stage for BIPHASE versus POLYPHASE evidence.")


if __name__ == "__main__":
    main()
