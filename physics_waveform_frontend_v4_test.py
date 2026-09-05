from collections import defaultdict

import numpy as np

from physics_waveform_frontend_v4 import PhysicsWaveformFrontendV4
from variable_bandwidth_waveform_test import clean_waveform
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def nlfm(num_samples, normalized_bandwidth=0.30):
    n = np.arange(num_samples, dtype=float)
    t = (n - 0.5 * (num_samples - 1)) / max(num_samples - 1, 1)
    # Cubic phase gives a smooth quadratic instantaneous-frequency law.
    phase = 2.0 * np.pi * normalized_bandwidth * num_samples * (t ** 3) / 3.0
    return np.exp(1j * phase)


def stepped_frequency(num_samples, steps=7, normalized_step=0.045):
    n = np.arange(num_samples, dtype=int)
    state = np.minimum(n * steps // num_samples, steps - 1)
    frequency = (state - 0.5 * (steps - 1)) * normalized_step
    phase = 2.0 * np.pi * np.cumsum(frequency)
    return np.exp(1j * phase)


def random_polyphase(num_samples, rng, chips=24):
    phases = rng.integers(0, 4, chips) * (0.5 * np.pi)
    idx = np.minimum(np.arange(num_samples) * chips // num_samples, chips - 1)
    return np.exp(1j * phases[idx])


def main():
    classifier = PhysicsWaveformFrontendV4(FS)
    names = ("CW", "LFM", "NLFM", "FREQUENCY_CODED", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE", "RANDOM_POLYPHASE")
    truth = {
        "CW": "CW",
        "LFM": "LFM",
        "NLFM": "NLFM",
        "FREQUENCY_CODED": "FREQUENCY_CODED",
        "BARKER13": "PHASE_CODED",
        "GENERIC_BIPHASE": "PHASE_CODED",
        "FRANK16_LIKE": "PHASE_CODED",
        "RANDOM_POLYPHASE": "PHASE_CODED",
    }
    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    trials = 20
    seed = 440000

    by_name = defaultdict(lambda: [0, 0])
    by_snr = defaultdict(lambda: [0, 0])
    confusion = defaultdict(lambda: defaultdict(int))

    for name in names:
        for width in widths_us:
            count = int(round(width * 1e-6 * FS))
            for snr in snrs_db:
                for _ in range(trials):
                    seed += 1
                    rng = np.random.default_rng(seed)
                    if name == "CW":
                        clean = clean_waveform("CW", count, 0.30)
                    elif name == "LFM":
                        clean = clean_waveform("LFM", count, rng.choice((0.15, 0.25, 0.35)))
                    elif name == "NLFM":
                        clean = nlfm(count, rng.choice((0.18, 0.28, 0.38)))
                    elif name == "FREQUENCY_CODED":
                        clean = stepped_frequency(count, int(rng.choice((5, 7, 9))), rng.choice((0.03, 0.045, 0.06)))
                    elif name == "RANDOM_POLYPHASE":
                        clean = random_polyphase(count, rng, int(rng.choice((16, 24, 32))))
                    else:
                        clean = clean_waveform(name, count, 0.30)

                    obs = impair(clean, snr, rng.uniform(-5e6, 5e6), rng.uniform(-np.pi, np.pi), int(rng.integers(-3, 4)), seed)
                    result = classifier.classify(obs)
                    correct = result.family == truth[name]
                    by_name[name][0] += int(correct)
                    by_name[name][1] += 1
                    by_snr[(name, snr)][0] += int(correct)
                    by_snr[(name, snr)][1] += 1
                    confusion[truth[name]][result.family] += 1

    print("S2B PHYSICS WAVEFORM FRONT END V4")
    print("=================================")
    print("Broad physical family recognition before Barker/Frank subtype analysis.")
    print()
    print("Overall accuracy by waveform")
    print("----------------------------")
    for name in names:
        c, t = by_name[name]
        print(f"{name:20s}{100*c/t:8.2f}%")

    print()
    print("Accuracy by SNR")
    print("---------------")
    print("waveform             0dB       5dB      10dB      15dB      20dB")
    for name in names:
        values = []
        for snr in snrs_db:
            c, t = by_snr[(name, snr)]
            values.append(100*c/t)
        print(f"{name:20s}" + "".join(f"{v:9.2f}%" for v in values))

    families = ("CW", "LFM", "NLFM", "FREQUENCY_CODED", "PHASE_CODED", "UNKNOWN")
    print()
    print("Broad-family confusion")
    print("----------------------")
    print("truth\\pred" + "".join(f"{f:>18s}" for f in families))
    for actual in families[:-1]:
        print(f"{actual:10s}" + "".join(f"{confusion[actual][pred]:18d}" for pred in families))

    total_correct = sum(v[0] for v in by_name.values())
    total = sum(v[1] for v in by_name.values())
    print()
    print(f"Overall broad-family accuracy: {100*total_correct/total:.2f}% ({total_correct}/{total})")
    print()
    print("RANDOM_POLYPHASE is deliberately expected to be PHASE_CODED, not UNKNOWN.")
    print("UNKNOWN is now reserved for pulses that do not adequately match a broad physical family.")


if __name__ == "__main__":
    main()
