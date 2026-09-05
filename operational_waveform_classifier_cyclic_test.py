from collections import defaultdict

import numpy as np

from operational_waveform_classifier import OperationalWaveformClassifier
from operational_waveform_classifier_cyclic import CyclicOperationalWaveformClassifier
from operational_waveform_classifier_test import make_clean
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def main():
    base = OperationalWaveformClassifier(FS)
    cyclic = CyclicOperationalWaveformClassifier(FS)

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
    )

    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    trials = 20
    seed = 920000
    results = defaultdict(lambda: [0, 0, 0])
    unknown_by_snr = defaultdict(lambda: [0, 0])

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
                    rb = base.classify(obs)
                    rc = cyclic.classify(obs)
                    results[name][0] += int(rb.family == truth)
                    results[name][1] += int(rc.family == truth)
                    results[name][2] += 1
                    if name == "DUAL_TONE":
                        unknown_by_snr[snr][0] += int(rc.family == "UNKNOWN")
                        unknown_by_snr[snr][1] += 1

    print("S2B CYCLIC-EVIDENCE CLASSIFIER COMPARISON")
    print("=========================================")
    print("The base classifier is unchanged. The experimental classifier adds only")
    print("lightweight x^2/x^4/envelope cyclic evidence.")
    print()
    print("waveform                base      cyclic      change")
    print("----------------------------------------------------")
    base_total = cyclic_total = total = 0
    for name, _ in cases:
        b, c, n = results[name]
        bp = 100.0 * b / n
        cp = 100.0 * c / n
        print(f"{name:20s}{bp:8.2f}% {cp:10.2f}% {cp-bp:+9.2f}")
        base_total += b
        cyclic_total += c
        total += n

    print()
    print(f"Overall base accuracy   : {100.0*base_total/total:.2f}%")
    print(f"Overall cyclic accuracy : {100.0*cyclic_total/total:.2f}%")
    print()
    print("DUAL_TONE UNKNOWN rejection by SNR")
    print("----------------------------------")
    for snr in snrs_db:
        good, n = unknown_by_snr[snr]
        print(f"{snr:2d} dB : {100.0*good/n:7.2f}%")

    known_names = [name for name, truth in cases if truth != "UNKNOWN"]
    known_b = sum(results[name][0] for name in known_names)
    known_c = sum(results[name][1] for name in known_names)
    known_n = sum(results[name][2] for name in known_names)
    print()
    print(f"Known-family base accuracy   : {100.0*known_b/known_n:.2f}%")
    print(f"Known-family cyclic accuracy : {100.0*known_c/known_n:.2f}%")
    print()
    print("Decision criterion: useful if dual-tone UNKNOWN rejection improves strongly")
    print("without materially degrading CW/FM/PHASE_CODED accuracy.")


if __name__ == "__main__":
    main()
