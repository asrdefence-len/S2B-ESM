import numpy as np

from fft_pca_waveform_classifier import FFTPCAWaveformClassifier, default_clean_cases
from fft_pca_waveform_test import make_training_set
from waveform_classifier_stress_test import impair


SAMPLE_RATE_HZ = 40_000_000.0
PULSE_SAMPLES = 256
FAMILIES = ("CW", "LFM", "BIPHASE", "POLYPHASE")


def main():
    training_samples, training_labels, training_subtypes = make_training_set()
    classifier = FFTPCAWaveformClassifier(
        sample_rate_hz=SAMPLE_RATE_HZ,
        n_components=8,
    ).fit(training_samples, training_labels, training_subtypes)

    snrs_db = (0, 5, 10, 15, 20)
    trials = 100
    frequency_offset_hz = 1.7e6
    timing_shift = 3
    seed = 90000

    print("S2B FFT/PCA SOFT WAVEFORM EVIDENCE")
    print("==================================")
    print("Each row is the mean evidence vector over 100 independently impaired pulses.")
    print("Weights sum to one but are NOT calibrated probabilities.")
    print(f"Frequency offset : {frequency_offset_hz / 1e6:.1f} MHz")
    print(f"Timing shift     : {timing_shift:+d} samples")
    print()

    for truth_family, subtype, clean in default_clean_cases(PULSE_SAMPLES):
        display_name = subtype if subtype not in ("CW", "LFM") else truth_family
        print(f"Truth waveform: {display_name}  (family {truth_family})")
        print("SNR       CW       LFM    BIPHASE  POLYPHASE   top-family   hard-correct")

        for snr_db in snrs_db:
            accumulated = {family: 0.0 for family in FAMILIES}
            hard_correct = 0

            for _ in range(trials):
                seed += 1
                rng = np.random.default_rng(seed)
                phase_rad = rng.uniform(-np.pi, np.pi)
                observed = impair(
                    clean,
                    snr_db=snr_db,
                    frequency_offset_hz=frequency_offset_hz,
                    phase_rad=phase_rad,
                    timing_shift=timing_shift,
                    seed=seed,
                )
                result = classifier.classify(observed)
                for family in FAMILIES:
                    accumulated[family] += result.family_scores.get(family, 0.0)
                hard_correct += int(result.family == truth_family)

            means = {family: accumulated[family] / trials for family in FAMILIES}
            top = max(means, key=means.get)
            print(
                f"{snr_db:2d} dB  "
                f"{means['CW']:7.3f}  {means['LFM']:7.3f}  "
                f"{means['BIPHASE']:9.3f}  {means['POLYPHASE']:9.3f}   "
                f"{top:10s}   {hard_correct:3d}/{trials}"
            )
        print()

    print("Interpretation check")
    print("--------------------")
    print("Desired behaviour:")
    print("  * high-SNR evidence concentrates toward the correct family")
    print("  * low-SNR evidence becomes flatter instead of confidently wrong")
    print("  * coded-waveform ambiguity is mainly BIPHASE versus POLYPHASE")
    print()
    print("Do not integrate these weights into MHT until this behaviour is checked.")


if __name__ == "__main__":
    main()
