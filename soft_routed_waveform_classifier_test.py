from collections import defaultdict

import numpy as np

from hierarchical_waveform_classifier import HierarchicalWaveformClassifier
from soft_routed_waveform_classifier import SoftRoutedWaveformClassifier
from variable_bandwidth_waveform_test import clean_waveform, make_training_set, truth_family
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def main():
    training_samples, training_labels = make_training_set()
    hard = HierarchicalWaveformClassifier(FS).fit_coded(training_samples, training_labels)
    soft = SoftRoutedWaveformClassifier(FS).fit_coded(training_samples, training_labels)

    names = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")
    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    lfm_bandwidths = (0.15, 0.25, 0.35)
    trials = 40
    seed = 280000

    hard_by_name = defaultdict(lambda: [0, 0])
    soft_by_name = defaultdict(lambda: [0, 0])
    soft_by_snr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    confusion = defaultdict(lambda: defaultdict(int))
    coded_runs = 0
    total = 0

    families = ("CW", "LFM", "BIPHASE", "POLYPHASE", "UNKNOWN")

    for name in names:
        truth = truth_family(name)
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
                        h = hard.classify(observed)
                        s = soft.classify(observed)
                        hard_by_name[name][0] += int(h.family == truth)
                        hard_by_name[name][1] += 1
                        soft_by_name[name][0] += int(s.family == truth)
                        soft_by_name[name][1] += 1
                        soft_by_snr[name][snr_db][0] += int(s.family == truth)
                        soft_by_snr[name][snr_db][1] += 1
                        confusion[truth][s.family] += 1
                        coded_runs += int(s.coded_stage_run)
                        total += 1

    print("S2B HARD ROUTING vs SOFT ROUTING")
    print("================================")
    print("Same impaired pulses are classified by both architectures.")
    print()
    print("Overall accuracy by waveform")
    print("----------------------------")
    print("waveform              HARD        SOFT")
    for name in names:
        hg, ht = hard_by_name[name]
        sg, st = soft_by_name[name]
        print(f"{name:18s}{100.0*hg/ht:8.2f}%  {100.0*sg/st:8.2f}%")

    print()
    print("Soft-routing accuracy by SNR")
    print("----------------------------")
    print("waveform             0dB       5dB      10dB      15dB      20dB")
    for name in names:
        values = []
        for snr in snrs_db:
            good, count = soft_by_snr[name][snr]
            values.append(f"{100.0*good/count:8.2f}%")
        print(f"{name:18s}" + " ".join(values))

    print()
    print("Soft-routing family confusion matrix")
    print("------------------------------------")
    print("truth\\pred       " + " ".join(f"{family:>11s}" for family in families))
    for truth in ("CW", "LFM", "BIPHASE", "POLYPHASE"):
        values = " ".join(f"{confusion[truth][pred]:11d}" for pred in families)
        print(f"{truth:14s} {values}")

    hard_good = sum(v[0] for v in hard_by_name.values())
    hard_total = sum(v[1] for v in hard_by_name.values())
    soft_good = sum(v[0] for v in soft_by_name.values())
    soft_total = sum(v[1] for v in soft_by_name.values())
    print()
    print(f"Hard-routing overall : {100.0*hard_good/hard_total:.2f}% ({hard_good}/{hard_total})")
    print(f"Soft-routing overall : {100.0*soft_good/soft_total:.2f}% ({soft_good}/{soft_total})")
    print(f"Coded PCA invoked    : {100.0*coded_runs/total:.2f}% of pulses ({coded_runs}/{total})")
    print()
    print("The soft family scores are engineering evidence weights, not calibrated probabilities.")


if __name__ == "__main__":
    main()
