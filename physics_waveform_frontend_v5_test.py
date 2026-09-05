from collections import defaultdict

import numpy as np

from physics_waveform_frontend_v5 import PhysicsWaveformFrontendV5
from physics_waveform_frontend_v4_test import nlfm, stepped_frequency, random_polyphase
from variable_bandwidth_waveform_test import clean_waveform
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def main():
    classifier = PhysicsWaveformFrontendV5(FS)
    names = ("CW", "LFM", "NLFM", "FREQUENCY_CODED", "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE", "RANDOM_POLYPHASE")
    truth = {
        "CW": "CW", "LFM": "LFM", "NLFM": "NLFM", "FREQUENCY_CODED": "FREQUENCY_CODED",
        "BARKER13": "PHASE_CODED", "GENERIC_BIPHASE": "PHASE_CODED",
        "FRANK16_LIKE": "PHASE_CODED", "RANDOM_POLYPHASE": "PHASE_CODED",
    }
    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (0, 5, 10, 15, 20)
    trials = 20
    seed = 550000

    by_name = defaultdict(lambda: [0, 0])
    by_snr = defaultdict(lambda: [0, 0])
    confusion = defaultdict(lambda: defaultdict(int))
    diag = defaultdict(list)

    for name in names:
        for width in widths_us:
            count = int(round(width * 1e-6 * FS))
            for snr in snrs_db:
                for _ in range(trials):
                    seed += 1
                    rng = np.random.default_rng(seed)
                    if name == "CW": clean = clean_waveform("CW", count, 0.30)
                    elif name == "LFM": clean = clean_waveform("LFM", count, rng.choice((0.15, 0.25, 0.35)))
                    elif name == "NLFM": clean = nlfm(count, rng.choice((0.18, 0.28, 0.38)))
                    elif name == "FREQUENCY_CODED": clean = stepped_frequency(count, int(rng.choice((5, 7, 9))), rng.choice((0.03, 0.045, 0.06)))
                    elif name == "RANDOM_POLYPHASE": clean = random_polyphase(count, rng, int(rng.choice((16, 24, 32))))
                    else: clean = clean_waveform(name, count, 0.30)

                    obs = impair(clean, snr, rng.uniform(-5e6, 5e6), rng.uniform(-np.pi, np.pi), int(rng.integers(-3, 4)), seed)
                    r = classifier.classify(obs)
                    actual = truth[name]
                    correct = r.family == actual
                    by_name[name][0] += int(correct); by_name[name][1] += 1
                    by_snr[(name, snr)][0] += int(correct); by_snr[(name, snr)][1] += 1
                    confusion[actual][r.family] += 1
                    if snr >= 15 and name in ("CW", "LFM", "NLFM", "FREQUENCY_CODED"):
                        diag[name].append((r.excursion_hz, r.linear_rms_hz, r.quadratic_rms_hz, r.nlfm_improvement, r.jump_count, r.median_abs_df_hz))

    print("S2B PHYSICS WAVEFORM FRONT END V5")
    print("=================================")
    print("Instantaneous-frequency trajectory classification with phase-code routing.")
    print()
    print("Overall accuracy by waveform")
    print("----------------------------")
    for name in names:
        c, t = by_name[name]; print(f"{name:20s}{100*c/t:8.2f}%")

    print("\nAccuracy by SNR\n---------------")
    print("waveform             0dB       5dB      10dB      15dB      20dB")
    for name in names:
        vals = [100*by_snr[(name,s)][0]/by_snr[(name,s)][1] for s in snrs_db]
        print(f"{name:20s}" + "".join(f"{v:9.2f}%" for v in vals))

    families = ("CW", "LFM", "NLFM", "FREQUENCY_CODED", "PHASE_CODED", "UNKNOWN")
    print("\nBroad-family confusion\n----------------------")
    print("truth\\pred" + "".join(f"{f:>18s}" for f in families))
    for actual in families[:-1]:
        print(f"{actual:10s}" + "".join(f"{confusion[actual][p]:18d}" for p in families))

    print("\nHigh-SNR trajectory medians (15/20 dB)")
    print("--------------------------------------")
    print("waveform             excursion  linearRMS  quadRMS  NLFMgain  jumps  |df|")
    print("                     MHz        MHz        MHz                count  kHz")
    for name in ("CW", "LFM", "NLFM", "FREQUENCY_CODED"):
        a = np.asarray(diag[name], dtype=float)
        med = np.median(a, axis=0)
        print(f"{name:20s}{med[0]/1e6:8.3f}   {med[1]/1e6:8.3f}   {med[2]/1e6:8.3f}   {med[3]:8.3f}  {med[4]:5.1f}  {med[5]/1e3:7.2f}")

    total_correct = sum(v[0] for v in by_name.values()); total = sum(v[1] for v in by_name.values())
    print(f"\nOverall broad-family accuracy: {100*total_correct/total:.2f}% ({total_correct}/{total})")
    print("V4 remains untouched as the comparison baseline.")


if __name__ == "__main__":
    main()
