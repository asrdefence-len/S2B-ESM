import argparse
from collections import defaultdict

import numpy as np

from waveform_classifier import WaveformClassifier
from waveform_library import barker13, biphase, cw, lfm, polyphase


SAMPLE_RATE_HZ = 40_000_000.0
PULSE_SAMPLES = 256
FAMILIES = ("CW", "LFM", "BIPHASE", "POLYPHASE", "UNKNOWN")


def impair(samples, snr_db, frequency_offset_hz, phase_rad, timing_shift, seed):
    rng = np.random.default_rng(seed)
    clean = np.asarray(samples, dtype=np.complex128)

    if timing_shift:
        shifted = np.zeros_like(clean)
        if timing_shift > 0:
            shifted[timing_shift:] = clean[:-timing_shift]
        else:
            shift = -timing_shift
            shifted[:-shift] = clean[shift:]
        clean = shifted

    n = np.arange(len(clean), dtype=float)
    carrier = np.exp(
        1j * (2.0 * np.pi * frequency_offset_hz * n / SAMPLE_RATE_HZ + phase_rad)
    )
    signal = clean * carrier

    signal_power = max(float(np.mean(np.abs(signal) ** 2)), 1e-12)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return (signal + noise).astype(np.complex64)


def waveform_cases():
    return [
        ("CW", "CW", cw(PULSE_SAMPLES)),
        ("LFM", "LFM", lfm(PULSE_SAMPLES, 0.30)),
        ("BARKER13", "BIPHASE", barker13(PULSE_SAMPLES)),
        ("BIPHASE", "BIPHASE", biphase(PULSE_SAMPLES)),
        ("POLYPHASE", "POLYPHASE", polyphase(PULSE_SAMPLES)),
    ]


def main():
    parser = argparse.ArgumentParser(description="Stress-test the S2B waveform classifier.")
    parser.add_argument("--trials", type=int, default=20, help="Random trials per grid point")
    args = parser.parse_args()

    snrs_db = (0, 5, 10, 15, 20)
    offsets_hz = (-5e6, -2.5e6, 0.0, 2.5e6, 5e6)
    timing_shifts = (-3, 0, 3)

    classifier = WaveformClassifier(min_confidence=0.45)
    confusion = defaultdict(lambda: defaultdict(int))
    per_snr = defaultdict(lambda: [0, 0])
    per_waveform = defaultdict(lambda: [0, 0])
    per_timing = defaultdict(lambda: [0, 0])
    per_waveform_timing = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    total = 0
    correct = 0
    seed = 1000

    for name, truth_family, clean in waveform_cases():
        for snr_db in snrs_db:
            for offset_hz in offsets_hz:
                for timing_shift in timing_shifts:
                    for _ in range(args.trials):
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
                        confusion[truth_family][result.family] += 1
                        per_snr[snr_db][0] += int(ok)
                        per_snr[snr_db][1] += 1
                        per_waveform[name][0] += int(ok)
                        per_waveform[name][1] += 1
                        per_timing[timing_shift][0] += int(ok)
                        per_timing[timing_shift][1] += 1
                        per_waveform_timing[name][timing_shift][0] += int(ok)
                        per_waveform_timing[name][timing_shift][1] += 1

    print("S2B WAVEFORM CLASSIFIER STRESS TEST")
    print("===================================")
    print(f"Trials per grid point : {args.trials}")
    print(f"Total classifications : {total}")
    print("SNR grid              : 0, 5, 10, 15, 20 dB")
    print("Frequency offsets     : -5, -2.5, 0, +2.5, +5 MHz")
    print("Timing shifts         : -3, 0, +3 samples")
    print()

    print("Accuracy by waveform")
    print("--------------------")
    for name, _, _ in waveform_cases():
        good, count = per_waveform[name]
        print(f"{name:10s} {100.0 * good / count:6.2f}%  ({good}/{count})")

    print()
    print("Accuracy by SNR")
    print("---------------")
    for snr_db in snrs_db:
        good, count = per_snr[snr_db]
        print(f"{snr_db:2d} dB       {100.0 * good / count:6.2f}%  ({good}/{count})")

    print()
    print("Accuracy by timing shift")
    print("------------------------")
    for timing_shift in timing_shifts:
        good, count = per_timing[timing_shift]
        sign = "+" if timing_shift > 0 else ""
        print(
            f"{sign}{timing_shift:2d} samples  "
            f"{100.0 * good / count:6.2f}%  ({good}/{count})"
        )

    print()
    print("Waveform accuracy by timing shift")
    print("---------------------------------")
    header = "waveform   " + " ".join(
        f"{('+' if shift > 0 else '') + str(shift):>10s}" for shift in timing_shifts
    )
    print(header)
    for name, _, _ in waveform_cases():
        values = []
        for timing_shift in timing_shifts:
            good, count = per_waveform_timing[name][timing_shift]
            values.append(f"{100.0 * good / count:9.2f}%")
        print(f"{name:10s} " + " ".join(values))

    print()
    print("Family confusion matrix (counts)")
    print("--------------------------------")
    header = "truth\\pred " + " ".join(f"{family:>10s}" for family in FAMILIES)
    print(header)
    for truth in ("CW", "LFM", "BIPHASE", "POLYPHASE"):
        values = " ".join(f"{confusion[truth][pred]:10d}" for pred in FAMILIES)
        print(f"{truth:10s} {values}")

    print()
    print(f"Overall family accuracy : {100.0 * correct / total:.2f}% ({correct}/{total})")
    print()
    print("NOTE: this is a synthetic impairment test, not a calibrated probability or")
    print("a claim of operational classification performance. The classifier remains")
    print("separate from the operational PDW extractor until these limits are understood.")


if __name__ == "__main__":
    main()
