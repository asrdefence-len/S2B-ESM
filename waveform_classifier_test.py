import numpy as np

from waveform_classifier import WaveformClassifier
from waveform_library import barker13, biphase, cw, lfm, polyphase


SAMPLE_RATE_HZ = 40_000_000.0
PULSE_SAMPLES = 256


def impair(samples, snr_db=20.0, frequency_offset_hz=1.7e6, phase_rad=0.73, seed=1):
    rng = np.random.default_rng(seed)
    n = np.arange(len(samples), dtype=float)
    carrier = np.exp(1j * (2.0 * np.pi * frequency_offset_hz * n / SAMPLE_RATE_HZ + phase_rad))
    signal = samples.astype(np.complex128) * carrier

    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return (signal + noise).astype(np.complex64)


def main():
    classifier = WaveformClassifier(min_confidence=0.45)

    cases = [
        ("CW", cw(PULSE_SAMPLES)),
        ("LFM", lfm(PULSE_SAMPLES, 0.30)),
        ("BARKER13", barker13(PULSE_SAMPLES)),
        ("BIPHASE", biphase(PULSE_SAMPLES)),
        ("POLYPHASE", polyphase(PULSE_SAMPLES)),
    ]

    print("S2B WAVEFORM CLASSIFIER EXPERIMENT")
    print("==================================")
    print(f"Samples per pulse : {PULSE_SAMPLES}")
    print(f"Sample rate       : {SAMPLE_RATE_HZ / 1e6:.1f} MS/s")
    print("Impairments        : 20 dB SNR, 1.7 MHz offset, arbitrary phase")
    print()

    passed = 0
    for index, (truth, clean) in enumerate(cases, start=1):
        observed = impair(clean, seed=index)
        result = classifier.classify(observed)

        expected_family = {
            "CW": "CW",
            "LFM": "LFM",
            "BARKER13": "BIPHASE",
            "BIPHASE": "BIPHASE",
            "POLYPHASE": "POLYPHASE",
        }[truth]
        ok = result.family == expected_family
        passed += int(ok)

        print(
            f"{truth:10s} -> {result.family:9s} / {result.subtype:16s} "
            f"confidence={100.0 * result.confidence:5.1f}% "
            f"fit={result.score:.3f} "
            f"runner-up={result.runner_up_family}/{result.runner_up_subtype} "
            f"({result.runner_up_score:.3f})  {'PASS' if ok else 'FAIL'}"
        )

    print()
    print(f"Family classification : {passed} / {len(cases)} passed")
    print()
    print("NOTE: confidence is an engineering similarity measure, not a calibrated probability.")
    print("This experiment is intentionally separate from the operational PDW extractor for now.")


if __name__ == "__main__":
    main()
