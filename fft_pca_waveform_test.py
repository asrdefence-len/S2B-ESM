from collections import defaultdict

import numpy as np

from fft_pca_waveform_classifier import FFTPCAWaveformClassifier, default_clean_cases
from waveform_classifier_stress_test import impair


SAMPLE_RATE_HZ = 40_000_000.0
PULSE_SAMPLES = 256
FAMILIES = ("CW", "LFM", "BIPHASE", "POLYPHASE")


def make_training_set(trials_per_case=120):
    samples = []
    labels = []
    subtypes = []
    seed = 20000

    # Train over nuisance variables. Timing shifts are deliberately included:
    # FFT magnitude should learn waveform spectral shape rather than pulse-window
    # alignment. Test seeds are disjoint from training seeds.
    for family, subtype, clean in default_clean_cases(PULSE_SAMPLES):
        for _ in range(trials_per_case):
            seed += 1
            rng = np.random.default_rng(seed)
            snr_db = rng.uniform(5.0, 25.0)
            offset_hz = rng.uniform(-5e6, 5e6)
            phase_rad = rng.uniform(-np.pi, np.pi)
            timing_shift = int(rng.integers(-4, 5))
            samples.append(
                impair(clean, snr_db, offset_hz, phase_rad, timing_shift, seed)
            )
            labels.append(family)
            subtypes.append(subtype)
    return samples, labels, subtypes


def main():
    training_samples, training_labels, training_subtypes = make_training_set()
    classifier = FFTPCAWaveformClassifier(
        sample_rate_hz=SAMPLE_RATE_HZ,
        n_components=8,
    ).fit(training_samples, training_labels, training_subtypes)

    snrs_db = (0, 5, 10, 15, 20)
    offsets_hz = (-5e6, -2.5e6, 0.0, 2.5e6, 5e6)
    timing_shifts = (-3, 0, 3)
    trials = 20

    per_waveform_timing = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_snr = defaultdict(lambda: [0, 0])
    family_confusion = defaultdict(lambda: defaultdict(int))
    waveform_confusion = defaultdict(lambda: defaultdict(int))
    total = 0
    correct = 0
    seed = 50000

    print("S2B FFT-MAGNITUDE + PCA WAVEFORM EXPERIMENT")
    print("===========================================")
    print(f"Training examples    : {len(training_samples)}")
    print("PCA dimensions       : 8")
    print("Feature              : centred normalised log FFT power")
    print("Classifier           : nearest family centroid in whitened PCA space")
    print()

    for name_family, subtype, clean in default_clean_cases(PULSE_SAMPLES):
        display_name = subtype if subtype != "CW" and subtype != "LFM" else name_family
        truth_family = name_family
        for snr_db in snrs_db:
            for offset_hz in offsets_hz:
                for timing_shift in timing_shifts:
                    for _ in range(trials):
                        seed += 1
                        rng = np.random.default_rng(seed)
                        phase_rad = rng.uniform(-np.pi, np.pi)
                        observed = impair(
                            clean,
                            snr_db=snr_db,
                            frequency_offset_hz=offset_hz,
                            phase_rad=phase_rad,
                            timing_shift=timing_shift,
                            seed=seed,
                        )
                        result = classifier.classify(observed)
                        ok = result.family == truth_family
                        total += 1
                        correct += int(ok)
                        per_waveform_timing[display_name][timing_shift][0] += int(ok)
                        per_waveform_timing[display_name][timing_shift][1] += 1
                        per_snr[snr_db][0] += int(ok)
                        per_snr[snr_db][1] += 1
                        family_confusion[truth_family][result.family] += 1
                        waveform_confusion[display_name][result.family] += 1

    print("Waveform accuracy by timing shift")
    print("---------------------------------")
    print("waveform           -3          0         +3")
    names = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")
    for name in names:
        values = []
        for shift in timing_shifts:
            good, count = per_waveform_timing[name][shift]
            values.append(f"{100.0 * good / count:9.2f}%")
        print(f"{name:16s}" + " ".join(values))

    print()
    print("Accuracy by SNR")
    print("---------------")
    for snr_db in snrs_db:
        good, count = per_snr[snr_db]
        print(f"{snr_db:2d} dB       {100.0 * good / count:6.2f}%  ({good}/{count})")

    print()
    print("Family confusion matrix (counts)")
    print("--------------------------------")
    print("truth\\pred " + " ".join(f"{family:>10s}" for family in FAMILIES))
    for truth in FAMILIES:
        values = " ".join(f"{family_confusion[truth][pred]:10d}" for pred in FAMILIES)
        print(f"{truth:10s} {values}")

    print()
    print("Waveform-to-family confusion (counts)")
    print("------------------------------------")
    print("truth waveform      " + " ".join(f"{family:>10s}" for family in FAMILIES))
    for name in names:
        values = " ".join(f"{waveform_confusion[name][pred]:10d}" for pred in FAMILIES)
        print(f"{name:18s}" + values)

    print()
    print(f"Overall family accuracy : {100.0 * correct / total:.2f}% ({correct}/{total})")
    print()
    print("The waveform-level table separates Barker-13 from generic biphase even though")
    print("both have BIPHASE as their family truth. This shows where family errors go.")
    print("This remains an experimental classifier and is not integrated into PDW extraction.")


if __name__ == "__main__":
    main()
