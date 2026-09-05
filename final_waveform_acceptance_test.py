from collections import defaultdict

import numpy as np

from strict_operational_waveform_classifier import StrictOperationalWaveformClassifier
from operational_waveform_classifier_test import dual_tone
from physics_waveform_frontend_v4_test import nlfm, stepped_frequency, random_polyphase
from variable_bandwidth_waveform_test import clean_waveform
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def sinusoidal_fm(num_samples, deviation=0.12, cycles=2.5):
    n = np.arange(num_samples, dtype=float)
    t = n / max(num_samples - 1, 1)
    inst = deviation * np.sin(2.0 * np.pi * cycles * t)
    phase = 2.0 * np.pi * np.cumsum(inst)
    return np.exp(1j * phase)


def am_tone(num_samples, depth=0.65, cycles=5.0):
    n = np.arange(num_samples, dtype=float)
    t = n / max(num_samples - 1, 1)
    envelope = 1.0 + depth * np.cos(2.0 * np.pi * cycles * t)
    return envelope.astype(np.complex128)


def hybrid_fm_phase(num_samples):
    base = clean_waveform("LFM", num_samples, 0.25)
    phases = np.array([0.0, np.pi, 0.0, 0.0, np.pi, 0.0, np.pi, 0.0, 0.0, np.pi, 0.0])
    idx = np.minimum(np.arange(num_samples) * len(phases) // num_samples, len(phases) - 1)
    return base * np.exp(1j * phases[idx])


def make_waveform(name, count, rng):
    if name == "CW":
        return clean_waveform("CW", count, 0.30)
    if name == "LFM":
        return clean_waveform("LFM", count, 0.25)
    if name == "NLFM":
        return nlfm(count, 0.28)
    if name == "STEPPED_FREQ":
        return stepped_frequency(count, 7, 0.045)
    if name == "BARKER13":
        return clean_waveform("BARKER13", count, 0.30)
    if name == "GENERIC_BIPHASE":
        return clean_waveform("GENERIC_BIPHASE", count, 0.30)
    if name == "FRANK16_LIKE":
        return clean_waveform("FRANK16_LIKE", count, 0.30)
    if name == "DUAL_TONE":
        return dual_tone(count, 0.18)
    if name == "SINUSOIDAL_FM":
        return sinusoidal_fm(count, deviation=0.12, cycles=2.5)
    if name == "AM_TONE":
        return am_tone(count, depth=0.65, cycles=5.0)
    if name == "HYBRID_FM_PHASE":
        return hybrid_fm_phase(count)
    if name == "RANDOM_POLYPHASE":
        return random_polyphase(count, rng, 29)
    raise ValueError(name)


def main():
    print("Building known-waveform library...")
    classifier = StrictOperationalWaveformClassifier(FS)

    cases = (
        ("CW", "CW"),
        ("LFM", "FM"),
        ("NLFM", "FM"),
        ("STEPPED_FREQ", "FM"),
        ("BARKER13", "PHASE_CODED"),
        ("GENERIC_BIPHASE", "PHASE_CODED"),
        ("FRANK16_LIKE", "PHASE_CODED"),
        ("DUAL_TONE", "UNKNOWN"),
        ("SINUSOIDAL_FM", "UNKNOWN"),
        ("AM_TONE", "UNKNOWN"),
        ("HYBRID_FM_PHASE", "UNKNOWN"),
        ("RANDOM_POLYPHASE", "UNKNOWN"),
    )

    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (5, 10, 15, 20)
    trials = 20
    seed = 990000

    stats = defaultdict(lambda: [0, 0])
    by_snr = defaultdict(lambda: [0, 0])
    confusion = defaultdict(lambda: defaultdict(int))
    accepted_ids = defaultdict(lambda: defaultdict(int))

    for name, truth in cases:
        for width in widths_us:
            count = int(round(width * 1e-6 * FS))
            for snr in snrs_db:
                for _ in range(trials):
                    seed += 1
                    rng = np.random.default_rng(seed)
                    clean = make_waveform(name, count, rng)
                    observed = impair(
                        clean,
                        snr_db=snr,
                        frequency_offset_hz=rng.uniform(-5e6, 5e6),
                        phase_rad=rng.uniform(-np.pi, np.pi),
                        timing_shift=int(rng.integers(-3, 4)),
                        seed=seed,
                    )
                    result = classifier.classify(observed)
                    correct = result.family == truth
                    stats[name][0] += int(correct)
                    stats[name][1] += 1
                    by_snr[(name, snr)][0] += int(correct)
                    by_snr[(name, snr)][1] += 1
                    confusion[truth][result.family] += 1
                    if result.library_accepted:
                        accepted_ids[name][result.library_id] += 1

    print("S2B STRICT LIBRARY-GATED WAVEFORM ACCEPTANCE TEST")
    print("=================================================")
    print("Broad physics classification is followed by a known-waveform library gate.")
    print("A broad family match is not sufficient: out-of-library waveforms become UNKNOWN.")
    print()
    print("Overall accuracy by waveform")
    print("----------------------------")
    for name, truth in cases:
        good, total = stats[name]
        detail = ""
        if accepted_ids[name]:
            best_id, count = max(accepted_ids[name].items(), key=lambda item: item[1])
            detail = f"  common library match={best_id} ({count})"
        print(f"{name:20s} truth={truth:12s} {100.0*good/total:8.2f}%{detail}")

    print()
    print("Accuracy by SNR")
    print("---------------")
    print("waveform             5dB      10dB      15dB      20dB")
    for name, _ in cases:
        vals = []
        for snr in snrs_db:
            good, total = by_snr[(name, snr)]
            vals.append(100.0 * good / total)
        print(f"{name:20s}" + "".join(f"{v:9.2f}%" for v in vals))

    families = ("CW", "FM", "PHASE_CODED", "UNKNOWN")
    print()
    print("Confusion matrix")
    print("----------------")
    print("truth\\pred" + "".join(f"{family:>15s}" for family in families))
    for truth in families:
        print(f"{truth:14s}" + "".join(f"{confusion[truth][pred]:15d}" for pred in families))

    known_names = {"CW", "LFM", "NLFM", "STEPPED_FREQ", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE"}
    known_good = sum(stats[n][0] for n in known_names)
    known_total = sum(stats[n][1] for n in known_names)
    unknown_names = {n for n, t in cases if t == "UNKNOWN"}
    unknown_good = sum(stats[n][0] for n in unknown_names)
    unknown_total = sum(stats[n][1] for n in unknown_names)

    print()
    print(f"Known-waveform retention : {100.0*known_good/known_total:.2f}% ({known_good}/{known_total})")
    print(f"Unknown-waveform rejection: {100.0*unknown_good/unknown_total:.2f}% ({unknown_good}/{unknown_total})")
    print()
    print("Target: retain high known-waveform accuracy while strongly rejecting the unseen")
    print("hybrid and random-polyphase cases that the family-only classifier accepted.")


if __name__ == "__main__":
    main()
