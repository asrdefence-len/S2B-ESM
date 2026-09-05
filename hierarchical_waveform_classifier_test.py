from collections import defaultdict

import numpy as np

from hierarchical_waveform_classifier import HierarchicalWaveformClassifier
from variable_bandwidth_waveform_test import clean_waveform, make_training_set, truth_family
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def main():
    training_samples, training_labels = make_training_set()
    classifier = HierarchicalWaveformClassifier(FS).fit_coded(training_samples, training_labels)

    names = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")
    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    lfm_bandwidths = (0.15, 0.25, 0.35)
    trials = 40
    seed = 240000

    by_snr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_width = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    confusion = defaultdict(lambda: defaultdict(int))
    route_confusion = defaultdict(lambda: defaultdict(int))
    bw_values = defaultdict(list)

    families = ("CW", "LFM", "BIPHASE", "POLYPHASE", "UNKNOWN")
    routes = ("CW", "LFM", "PHASE_CODED", "UNKNOWN")

    for name in names:
        truth = truth_family(name)
        route_truth = "PHASE_CODED" if truth in ("BIPHASE", "POLYPHASE") else truth
        for width_us in widths_us:
            num_samples = int(round(width_us * 1e-6 * FS))
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
                            frequency_offset_hz=rng.uniform(-5e6, 5e6),
                            phase_rad=rng.uniform(-np.pi, np.pi),
                            timing_shift=int(rng.integers(-3, 4)),
                            seed=seed,
                        )
                        result = classifier.classify(observed)
                        ok = result.family == truth
                        by_snr[name][snr_db][0] += int(ok)
                        by_snr[name][snr_db][1] += 1
                        by_width[name][width_us][0] += int(ok)
                        by_width[name][width_us][1] += 1
                        confusion[truth][result.family] += 1
                        route_confusion[route_truth][result.coarse_family] += 1
                        if snr_db >= 15:
                            bw_values[(name, width_us)].append(result.occupied_bandwidth_hz)

    print("S2B HIERARCHICAL WAVEFORM CLASSIFIER TEST")
    print("=========================================")
    print("Stage 1 : V3 physics router -> CW / LFM / PHASE_CODED")
    print("Stage 2 : PHASE_CODED -> bandwidth-normalized FFT/PCA -> BIPHASE / POLYPHASE")
    print("Subtype : deliberately not claimed yet")
    print()
    print("End-to-end family accuracy by SNR")
    print("---------------------------------")
    print("waveform             0dB       5dB      10dB      15dB      20dB")
    for name in names:
        values = []
        for snr_db in snrs_db:
            good, total = by_snr[name][snr_db]
            values.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(values))

    print()
    print("End-to-end family accuracy by pulse width")
    print("-----------------------------------------")
    print("waveform            2.5us      4.5us      7.0us      9.0us")
    for name in names:
        values = []
        for width in widths_us:
            good, total = by_width[name][width]
            values.append(f"{100.0 * good / total:8.2f}%")
        print(f"{name:18s}" + " ".join(values))

    print()
    print("End-to-end family confusion matrix")
    print("----------------------------------")
    print("truth\\pred       " + " ".join(f"{family:>11s}" for family in families))
    for truth in ("CW", "LFM", "BIPHASE", "POLYPHASE"):
        values = " ".join(f"{confusion[truth][pred]:11d}" for pred in families)
        print(f"{truth:14s} {values}")

    print()
    print("Stage-1 route confusion matrix")
    print("------------------------------")
    print("truth\\route      " + " ".join(f"{route:>12s}" for route in routes))
    for truth in ("CW", "LFM", "PHASE_CODED"):
        values = " ".join(f"{route_confusion[truth][pred]:12d}" for pred in routes)
        print(f"{truth:14s} {values}")

    print()
    print("Representative occupied bandwidth at >=15 dB SNR")
    print("-------------------------------------------------")
    for name in names:
        values = []
        for width in widths_us:
            bw = np.mean(bw_values[(name, width)]) / 1e6
            values.append(f"{bw:6.2f} MHz")
        print(f"{name:18s}" + "  ".join(values))

    good = sum(v[0] for item in by_snr.values() for v in item.values())
    total = sum(v[1] for item in by_snr.values() for v in item.values())
    print()
    print(f"Overall end-to-end family accuracy : {100.0 * good / total:.2f}% ({good}/{total})")
    print()
    print("Interpret low-SNR failures as uncertainty evidence, not a reason to force a hard label.")


if __name__ == "__main__":
    main()
