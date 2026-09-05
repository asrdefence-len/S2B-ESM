from pathlib import Path
import tempfile

import numpy as np

from fft_pca_waveform_classifier import FFTPCAWaveformClassifier, default_clean_cases
from fft_pca_waveform_test import SAMPLE_RATE_HZ, make_training_set
from waveform_classifier_stress_test import impair


def result_vector(result):
    families = ("CW", "LFM", "BIPHASE", "POLYPHASE")
    return np.array([result.family_scores.get(family, 0.0) for family in families])


def main():
    training_samples, training_labels, training_subtypes = make_training_set()
    trained = FFTPCAWaveformClassifier(
        sample_rate_hz=SAMPLE_RATE_HZ,
        n_components=8,
    ).fit(training_samples, training_labels, training_subtypes)

    test_pulses = []
    seed = 81000
    for _, _, clean in default_clean_cases(256):
        for snr_db in (5, 10, 20):
            seed += 1
            test_pulses.append(
                impair(
                    clean,
                    snr_db=snr_db,
                    frequency_offset_hz=1.7e6,
                    phase_rad=0.73,
                    timing_shift=3,
                    seed=seed,
                )
            )

    before = [trained.classify(pulse) for pulse in test_pulses]

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "waveform_fft_pca_v1.npz"
        trained.save_model(
            model_path,
            metadata={
                "model_name": "waveform_fft_pca_v1",
                "training_examples": len(training_samples),
                "purpose": "S2B experimental waveform family classifier",
            },
        )
        loaded = FFTPCAWaveformClassifier.load_model(model_path)
        after = [loaded.classify(pulse) for pulse in test_pulses]

        print("S2B FFT/PCA MODEL SAVE-LOAD ROUNDTRIP TEST")
        print("==========================================")
        print(f"Training examples : {len(training_samples)}")
        print(f"Test pulses       : {len(test_pulses)}")
        print(f"Saved model       : {model_path.name}")
        print(f"Model format      : {loaded.model_metadata_['model_format_version']}")
        print(f"Feature           : {loaded.model_metadata_['feature_name']}")
        print(f"Sample rate       : {loaded.sample_rate_hz / 1e6:.1f} MHz")
        print(f"PCA dimensions    : {loaded.n_components}")
        print()

        max_score_error = 0.0
        all_identical = True
        for index, (a, b) in enumerate(zip(before, after), start=1):
            score_error = float(np.max(np.abs(result_vector(a) - result_vector(b))))
            max_score_error = max(max_score_error, score_error)
            same = (
                a.family == b.family
                and a.subtype == b.subtype
                and a.runner_up_family == b.runner_up_family
                and np.isclose(a.distance, b.distance, rtol=0.0, atol=1e-12)
                and np.isclose(a.confidence, b.confidence, rtol=0.0, atol=1e-12)
                and score_error <= 1e-12
            )
            all_identical &= same
            print(
                f"Pulse {index:2d}: {a.family:9s} -> {b.family:9s}  "
                f"max evidence delta={score_error:.3e}  {'PASS' if same else 'FAIL'}"
            )

        print()
        print(f"Maximum evidence delta : {max_score_error:.3e}")
        print(f"Roundtrip result       : {'PASS' if all_identical else 'FAIL'}")

        if not all_identical:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
