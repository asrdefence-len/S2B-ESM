from collections import defaultdict

import numpy as np

from open_set_waveform_classifier_v2 import OpenSetWaveformClassifierV2
from open_set_waveform_classifier_test import unseen_waveform
from variable_bandwidth_waveform_test import clean_waveform, make_training_set, truth_family
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def main():
    training_samples, training_labels = make_training_set()
    classifier = OpenSetWaveformClassifierV2(FS).fit(training_samples, training_labels)

    known = ("CW", "LFM", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE")
    unseen = ("NLFM", "STEPPED_FREQ", "RANDOM_PHASE32", "QPSK_CODE20")
    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (5, 10, 15, 20)
    trials = 30
    seed = 390000

    known_stats = defaultdict(lambda: [0, 0, 0])
    unknown_stats = defaultdict(lambda: [0, 0])
    false_labels = defaultdict(lambda: defaultdict(int))
    reject_reasons = defaultdict(lambda: defaultdict(int))
    adequacy = defaultdict(list)

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
                        r = classifier.classify(obs)
                        known_stats[name][0] += int(r.family == truth)
                        known_stats[name][1] += int(r.family == "UNKNOWN")
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
                    r = classifier.classify(obs)
                    unknown_stats[name][0] += int(r.family == "UNKNOWN")
                    unknown_stats[name][1] += 1
                    if r.family != "UNKNOWN":
                        false_labels[name][r.family] += 1
                    else:
                        reject_reasons[name][r.rejection_reason] += 1
                    adequacy[name].append(max(r.smooth_residual_ratio, r.coded_reconstruction_ratio, r.coded_nearest_ratio))

    print("S2B OPEN-SET WAVEFORM CLASSIFIER V2 TEST")
    print("========================================")
    print("V2 tests whether a known model adequately explains the pulse before accepting its label.")
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

    print()
    print("Dominant rejection reason")
    print("-------------------------")
    for name in unseen:
        reasons = reject_reasons[name]
        reason = max(reasons, key=reasons.get) if reasons else "none"
        count = reasons.get(reason, 0)
        print(f"{name:18s}{reason:42s} {count}")

    print()
    print("Median adequacy ratio for unseen waveforms (>1 means outside learned envelope)")
    print("----------------------------------------------------------------------------")
    for name in unseen:
        print(f"{name:18s}{np.median(adequacy[name]):8.3f}")

    known_correct = sum(v[0] for v in known_stats.values())
    known_total = sum(v[2] for v in known_stats.values())
    unknown_correct = sum(v[0] for v in unknown_stats.values())
    unknown_total = sum(v[1] for v in unknown_stats.values())
    print()
    print(f"Known classification accuracy : {100*known_correct/known_total:.2f}% ({known_correct}/{known_total})")
    print(f"Unknown rejection accuracy    : {100*unknown_correct/unknown_total:.2f}% ({unknown_correct}/{unknown_total})")
    print()
    print("Do not tune thresholds until the adequacy ratios show whether the models actually separate known from unseen pulses.")


if __name__ == "__main__":
    main()
