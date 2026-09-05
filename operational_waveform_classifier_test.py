from collections import defaultdict

import numpy as np

from operational_waveform_classifier import OperationalWaveformClassifier
from physics_waveform_frontend_v4_test import nlfm, stepped_frequency, random_polyphase
from variable_bandwidth_waveform_test import clean_waveform
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def dual_tone(num_samples, separation=0.18):
    n = np.arange(num_samples, dtype=float)
    return (
        np.exp(1j * 2.0 * np.pi * (-0.5 * separation) * n)
        + np.exp(1j * 2.0 * np.pi * (+0.5 * separation) * n)
    ) / np.sqrt(2.0)


def hybrid_fm_phase(num_samples):
    base = clean_waveform("LFM", num_samples, 0.25)
    chips = 11
    phases = np.where((np.arange(chips) % 3) == 0, np.pi, 0.0)
    idx = np.minimum(np.arange(num_samples) * chips // num_samples, chips - 1)
    return base * np.exp(1j * phases[idx])


def make_clean(name, count, rng):
    if name == "CW":
        return clean_waveform("CW", count, 0.30)
    if name == "LFM":
        return clean_waveform("LFM", count, rng.choice((0.15, 0.25, 0.35)))
    if name == "NLFM":
        return nlfm(count, rng.choice((0.18, 0.28, 0.38)))
    if name == "STEPPED_FREQ":
        return stepped_frequency(count, int(rng.choice((5, 7, 9))), rng.choice((0.03, 0.045, 0.06)))
    if name == "BARKER13":
        return clean_waveform("BARKER13", count, 0.30)
    if name == "GENERIC_BIPHASE":
        return clean_waveform("GENERIC_BIPHASE", count, 0.30)
    if name == "FRANK16_LIKE":
        return clean_waveform("FRANK16_LIKE", count, 0.30)
    if name == "RANDOM_POLYPHASE":
        return random_polyphase(count, rng, int(rng.choice((16, 24, 32))))
    if name == "DUAL_TONE":
        return dual_tone(count, rng.choice((0.12, 0.18, 0.24)))
    if name == "HYBRID_FM_PHASE":
        return hybrid_fm_phase(count)
    raise ValueError(name)


def main():
    classifier = OperationalWaveformClassifier(FS)

    cases = (
        ("CW", "CW"),
        ("LFM", "FM"),
        ("NLFM", "FM"),
        ("STEPPED_FREQ", "FM"),
        ("BARKER13", "PHASE_CODED"),
        ("GENERIC_BIPHASE", "PHASE_CODED"),
        ("FRANK16_LIKE", "PHASE_CODED"),
        ("RANDOM_POLYPHASE", "PHASE_CODED"),
        ("DUAL_TONE", "UNKNOWN"),
        ("HYBRID_FM_PHASE", "UNKNOWN"),
    )

    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    trials = 20
    seed = 810000

    by_case = defaultdict(lambda: [0, 0])
    by_snr = defaultdict(lambda: [0, 0])
    confusion = defaultdict(lambda: defaultdict(int))
    score_medians = defaultdict(list)

    for name, truth in cases:
        for width in widths_us:
            count = int(round(width * 1e-6 * FS))
            for snr in snrs_db:
                for _ in range(trials):
                    seed += 1
                    rng = np.random.default_rng(seed)
                    clean = make_clean(name, count, rng)
                    obs = impair(
                        clean,
                        snr_db=snr,
                        frequency_offset_hz=rng.uniform(-5e6, 5e6),
                        phase_rad=rng.uniform(-np.pi, np.pi),
                        timing_shift=int(rng.integers(-3, 4)),
                        seed=seed,
                    )
                    result = classifier.classify(obs)
                    correct = result.family == truth
                    by_case[name][0] += int(correct)
                    by_case[name][1] += 1
                    by_snr[(name, snr)][0] += int(correct)
                    by_snr[(name, snr)][1] += 1
                    confusion[truth][result.family] += 1
                    if snr >= 15:
                        score_medians[name].append((
                            result.scores.get("CW", 0.0),
                            result.scores.get("FM", 0.0),
                            result.scores.get("PHASE_CODED", 0.0),
                        ))

    print("S2B OPERATIONAL WAVEFORM CLASSIFIER TEST")
    print("========================================")
    print("Supported broad classes: CW / FM / PHASE_CODED / UNKNOWN")
    print("LFM, NLFM and stepped-frequency are intentionally all FM.")
    print("Barker, generic biphase, Frank-like and random polyphase are all PHASE_CODED.")
    print()

    print("Overall accuracy by test waveform")
    print("---------------------------------")
    for name, truth in cases:
        good, total = by_case[name]
        print(f"{name:20s} truth={truth:12s}  {100.0*good/total:7.2f}%")

    print()
    print("Accuracy by SNR")
    print("---------------")
    print("waveform             0dB       5dB      10dB      15dB      20dB")
    for name, _ in cases:
        values = []
        for snr in snrs_db:
            good, total = by_snr[(name, snr)]
            values.append(100.0 * good / total)
        print(f"{name:20s}" + "".join(f"{v:9.2f}%" for v in values))

    families = ("CW", "FM", "PHASE_CODED", "UNKNOWN")
    print()
    print("Broad-family confusion")
    print("----------------------")
    print("truth\\pred" + "".join(f"{family:>15s}" for family in families))
    for truth in families:
        print(f"{truth:14s}" + "".join(f"{confusion[truth][pred]:15d}" for pred in families))

    print()
    print("High-SNR median evidence scores (15/20 dB)")
    print("------------------------------------------")
    print("waveform                 CW       FM   PHASE_CODED")
    for name, _ in cases:
        values = np.asarray(score_medians[name], dtype=float)
        med = np.median(values, axis=0)
        print(f"{name:20s}{med[0]:8.3f} {med[1]:8.3f} {med[2]:12.3f}")

    total_good = sum(v[0] for v in by_case.values())
    total = sum(v[1] for v in by_case.values())
    print()
    print(f"Overall classifier accuracy : {100.0*total_good/total:.2f}% ({total_good}/{total})")
    print()
    print("DUAL_TONE and HYBRID_FM_PHASE are deliberately unsupported stress cases.")
    print("If they are forced into a supported class, UNKNOWN rejection still needs work.")


if __name__ == "__main__":
    main()
