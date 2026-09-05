from collections import defaultdict

import numpy as np

from bandwidth_normalized_fft_pca_classifier import BandwidthNormalizedFFTPCAClassifier
from waveform_classifier_stress_test import impair
from waveform_library import barker13, biphase, cw, lfm, polyphase


FS = 40_000_000.0
FAMILIES = ("CW", "LFM", "BIPHASE", "POLYPHASE")


def clean_waveform(name, num_samples, lfm_bw=0.30):
    if name == "CW":
        return cw(num_samples)
    if name == "LFM":
        return lfm(num_samples, lfm_bw)
    if name == "BARKER13":
        return barker13(num_samples)
    if name == "GENERIC_BIPHASE":
        return biphase(num_samples)
    if name == "FRANK16_LIKE":
        return polyphase(num_samples)
    raise ValueError(name)


def truth_family(name):
    if name in ("BARKER13", "GENERIC_BIPHASE"):
        return "BIPHASE"
    if name == "FRANK16_LIKE":
        return "POLYPHASE"
    return name


def make_training_set(trials_per_case=35):
    names = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")
    pulse_widths_us = (2.0, 3.5, 5.0, 6.5, 8.0, 10.0)
    lfm_bandwidths = (0.10, 0.20, 0.30, 0.40)
    samples = []
    labels = []
    seed = 120000

    for name in names:
        for pulse_width_us in pulse_widths_us:
            num_samples = max(16, int(round(pulse_width_us * 1e-6 * FS)))
            bandwidths = lfm_bandwidths if name == "LFM" else (0.30,)
            for lfm_bw in bandwidths:
                clean = clean_waveform(name, num_samples, lfm_bw)
                for _ in range(trials_per_case):
                    seed += 1
                    rng = np.random.default_rng(seed)
                    observed = impair(
                        clean,
                        snr_db=rng.uniform(5.0, 25.0),
                        frequency_offset_hz=rng.uniform(-5e6, 5e6),
                        phase_rad=rng.uniform(-np.pi, np.pi),
                        timing_shift=int(rng.integers(-3, 4)),
                        seed=seed,
                    )
                    samples.append(observed)
                    labels.append(truth_family(name))
    return samples, labels


def main():
    training_samples, training_labels = make_training_set()
    classifier = BandwidthNormalizedFFTPCAClassifier(
        sample_rate_hz=FS,
        feature_bins=256,
        n_components=8,
    ).fit(training_samples, training_labels)

    names = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")
    test_widths_us = (2.5, 4.5, 7.0, 9.0)  # deliberately between training widths
    snrs_db = (5, 10, 15, 20)
    lfm_bandwidths = (0.15, 0.25, 0.35)    # deliberately between training values
    trials = 20
    seed = 160000

    results = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    measured_bw = defaultdict(list)

    for name in names:
        family = truth_family(name)
        for pulse_width_us in test_widths_us:
            num_samples = int(round(pulse_width_us * 1e-6 * FS))
            bandwidths = lfm_bandwidths if name == "LFM" else (0.30,)
            for lfm_bw in bandwidths:
                clean = clean_waveform(name, num_samples, lfm_bw)
                for snr_db in snrs_db:
                    for _ in range(trials):
                        seed += 1
                        rng = np.random.default_rng(seed)
                        observed = impair(
                            clean,
                            snr_db=snr_db,
                            frequency_offset_hz=rng.uniform(-4e6, 4e6),
                            phase_rad=rng.uniform(-np.pi, np.pi),
                            timing_shift=int(rng.integers(-3, 4)),
                            seed=seed,
                        )
                        result = classifier.classify(observed)
                        results[name][pulse_width_us][0] += int(result.family == family)
                        results[name][pulse_width_us][1] += 1
                        if snr_db >= 15:
                            measured_bw[(name, pulse_width_us)].append(result.occupied_bandwidth_hz)

    print("S2B VARIABLE-BANDWIDTH WAVEFORM FAMILY EXPERIMENT")
    print("=================================================")
    print(f"Training examples : {len(training_samples)}")
    print("Training PW       : 2.0, 3.5, 5.0, 6.5, 8.0, 10.0 us")
    print("Test PW           : 2.5, 4.5, 7.0, 9.0 us (unseen during training)")
    print("Feature           : occupied-BW-normalized log FFT power -> PCA")
    print("Absolute BW       : retained separately from normalized family feature")
    print()
    print("Family accuracy by pulse width")
    print("------------------------------")
    print("waveform            2.5us      4.5us      7.0us      9.0us")
    for name in names:
        values = []
        for width in test_widths_us:
            good, total = results[name][width]
            values.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(values))

    print()
    print("Representative measured occupied bandwidth at >=15 dB SNR")
    print("----------------------------------------------------------")
    for name in names:
        values = []
        for width in test_widths_us:
            bw = np.mean(measured_bw[(name, width)]) / 1e6
            values.append(f"{bw:6.2f} MHz")
        print(f"{name:18s}" + "  ".join(values))

    total_good = sum(value[0] for by_width in results.values() for value in by_width.values())
    total_count = sum(value[1] for by_width in results.values() for value in by_width.values())
    print()
    print(f"Overall family accuracy : {100.0 * total_good / total_count:.2f}% ({total_good}/{total_count})")
    print()
    print("This test deliberately uses pulse widths and LFM bandwidths not present in training.")
    print("It asks whether normalized spectral shape generalizes while absolute bandwidth remains observable.")


if __name__ == "__main__":
    main()
