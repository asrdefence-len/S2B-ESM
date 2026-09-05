from collections import defaultdict

import numpy as np

from open_set_waveform_classifier import OpenSetWaveformClassifier
from variable_bandwidth_waveform_test import clean_waveform, make_training_set, truth_family
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def _chips(phases, num_samples):
    indices = np.minimum((np.arange(num_samples) * len(phases) // num_samples), len(phases) - 1)
    return np.exp(1j * np.asarray(phases)[indices])


def unseen_waveform(name, num_samples, rng):
    n = np.arange(num_samples, dtype=float)
    if name == "NLFM":
        # Quadratic instantaneous-frequency law: deliberately outside trained LFM.
        t = (n - 0.5 * (num_samples - 1)) / max(num_samples - 1, 1)
        phase = 2.0 * np.pi * 0.22 * num_samples * (t ** 3) / 3.0
        return np.exp(1j * phase)
    if name == "STEPPED_FREQ":
        steps = 7
        index = np.minimum(n.astype(int) * steps // num_samples, steps - 1)
        freq = (index - 0.5 * (steps - 1)) * 0.045
        phase = 2.0 * np.pi * np.cumsum(freq)
        return np.exp(1j * phase)
    if name == "RANDOM_PHASE32":
        phases = rng.uniform(-np.pi, np.pi, 32)
        return _chips(phases, num_samples)
    if name == "QPSK_CODE20":
        phases = rng.integers(0, 4, 20) * (0.5 * np.pi)
        return _chips(phases, num_samples)
    raise ValueError(name)


def main():
    training_samples, training_labels = make_training_set()
    classifier = OpenSetWaveformClassifier(FS).fit_coded(training_samples, training_labels)

    known = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")
    unseen = ("NLFM", "STEPPED_FREQ", "RANDOM_PHASE32", "QPSK_CODE20")
    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (5, 10, 15, 20)
    trials = 30
    seed = 330000

    known_stats = defaultdict(lambda: [0, 0, 0])  # correct, unknown, total
    unknown_stats = defaultdict(lambda: [0, 0])   # rejected, total
    false_labels = defaultdict(lambda: defaultdict(int))

    for name in known:
        truth = truth_family(name)
        for width in widths_us:
            count = int(round(width * 1e-6 * FS))
            bws = (0.15, 0.25, 0.35) if name == "LFM" else (0.30,)
            for bw in bws:
                clean = clean_waveform(name, count, bw)
                for snr in snrs_db:
                    for _ in range(trials):
                        seed += 1
                        rng = np.random.default_rng(seed)
                        obs = impair(clean, snr, rng.uniform(-5e6, 5e6), rng.uniform(-np.pi, np.pi), int(rng.integers(-3, 4)), seed)
                        result = classifier.classify(obs)
                        known_stats[name][0] += int(result.family == truth)
                        known_stats[name][1] += int(result.family == "UNKNOWN")
                        known_stats[name][2] += 1

    for name in unseen:
        for width in widths_us:
            count = int(round(width * 1e-6 * FS))
            for snr in snrs_db:
                for _ in range(trials):
                    seed += 1
                    rng = np.random.default_rng(seed)
                    clean = unseen_waveform(name, count, rng)
                    obs = impair(clean, snr, rng.uniform(-5e6, 5e6), rng.uniform(-np.pi, np.pi), int(rng.integers(-3, 4)), seed)
                    result = classifier.classify(obs)
                    unknown_stats[name][0] += int(result.family == "UNKNOWN")
                    unknown_stats[name][1] += 1
                    if result.family != "UNKNOWN":
                        false_labels[name][result.family] += 1

    print("S2B OPEN-SET WAVEFORM CLASSIFIER TEST")
    print("=====================================")
    print("Known families should remain classified; deliberately unseen waveforms should be UNKNOWN.")
    print()
    print("Known-family retention")
    print("----------------------")
    print("waveform              correct    rejected")
    for name in known:
        correct, rejected, total = known_stats[name]
        print(f"{name:18s}{100*correct/total:8.2f}%  {100*rejected/total:8.2f}%")

    print()
    print("Unseen-waveform UNKNOWN rejection")
    print("---------------------------------")
    print("waveform              UNKNOWN     most common false label")
    for name in unseen:
        rejected, total = unknown_stats[name]
        errors = false_labels[name]
        false = max(errors, key=errors.get) if errors else "-"
        print(f"{name:18s}{100*rejected/total:8.2f}%    {false}")

    known_correct = sum(v[0] for v in known_stats.values())
    known_total = sum(v[2] for v in known_stats.values())
    unknown_correct = sum(v[0] for v in unknown_stats.values())
    unknown_total = sum(v[1] for v in unknown_stats.values())
    print()
    print(f"Known classification accuracy : {100*known_correct/known_total:.2f}% ({known_correct}/{known_total})")
    print(f"Unknown rejection accuracy    : {100*unknown_correct/unknown_total:.2f}% ({unknown_correct}/{unknown_total})")
    print()
    print("This is an open-set engineering experiment; rejection thresholds are not calibrated probabilities.")


if __name__ == "__main__":
    main()
