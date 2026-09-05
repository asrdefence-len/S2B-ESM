import numpy as np

from operational_waveform_classifier_test import dual_tone, hybrid_fm_phase
from physics_waveform_frontend_v4_test import nlfm, stepped_frequency, random_polyphase
from variable_bandwidth_waveform_test import clean_waveform
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


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
    if name == "RANDOM_POLYPHASE":
        return random_polyphase(count, rng, 24)
    if name == "DUAL_TONE":
        return dual_tone(count, 0.18)
    if name == "HYBRID_FM_PHASE":
        return hybrid_fm_phase(count)
    raise ValueError(name)


def spectral_concentration(y):
    y = np.asarray(y, dtype=np.complex128)
    if len(y) < 16:
        return 0.0, 0.0, 0.0
    y = y - np.mean(y)
    nfft = 1 << int(np.ceil(np.log2(max(len(y), 64))))
    window = np.hanning(len(y))
    p = np.abs(np.fft.fftshift(np.fft.fft(y * window, n=nfft))) ** 2
    total = float(np.sum(p))
    if total <= 1e-18:
        return 0.0, 0.0, 0.0
    p /= total
    peak = float(np.max(p))
    top4 = float(np.sum(np.sort(p)[-4:]))
    entropy = float(-np.sum(p * np.log(p + 1e-15)) / np.log(len(p)))
    return peak, top4, entropy


def cyclic_features(x):
    x = np.asarray(x, dtype=np.complex128)
    # Lightweight cyclostationary surrogates. x^2 is especially useful for
    # biphase/BPSK structure because pi phase reversals disappear. x^4 similarly
    # removes four-state phase rotations. |x|^2 exposes envelope/beating cycles.
    p2, t2, e2 = spectral_concentration(x ** 2)
    p4, t4, e4 = spectral_concentration(x ** 4)
    pe, te, ee = spectral_concentration(np.abs(x) ** 2)
    return {
        "x2_peak": p2,
        "x2_top4": t2,
        "x2_entropy": e2,
        "x4_peak": p4,
        "x4_top4": t4,
        "x4_entropy": e4,
        "env_peak": pe,
        "env_top4": te,
        "env_entropy": ee,
    }


def main():
    names = (
        "CW", "LFM", "NLFM", "STEPPED_FREQ",
        "BARKER13", "GENERIC_BIPHASE", "FRANK16_LIKE", "RANDOM_POLYPHASE",
        "DUAL_TONE", "HYBRID_FM_PHASE",
    )
    width_us = 7.0
    count = int(round(width_us * 1e-6 * FS))
    snrs = (20, 15, 10, 5)
    trials = 50
    seed = 910000

    print("S2B LIGHTWEIGHT CYCLOSTATIONARY WAVEFORM DIAGNOSTIC")
    print("===================================================")
    print("Features are spectra of x^2, x^4 and |x|^2; this is a cheap cyclic-feature")
    print("experiment, not a full spectral-correlation-function implementation.")
    print()

    for snr in snrs:
        print(f"SNR = {snr} dB")
        print("waveform              x2peak  x2top4  x2H    x4peak  x4top4  x4H    envPeak envTop4 envH")
        print("------------------------------------------------------------------------------------------------")
        for name in names:
            rows = []
            for _ in range(trials):
                seed += 1
                rng = np.random.default_rng(seed)
                clean = make_waveform(name, count, rng)
                obs = impair(
                    clean,
                    snr_db=snr,
                    frequency_offset_hz=rng.uniform(-5e6, 5e6),
                    phase_rad=rng.uniform(-np.pi, np.pi),
                    timing_shift=int(rng.integers(-3, 4)),
                    seed=seed,
                )
                f = cyclic_features(obs)
                rows.append((
                    f["x2_peak"], f["x2_top4"], f["x2_entropy"],
                    f["x4_peak"], f["x4_top4"], f["x4_entropy"],
                    f["env_peak"], f["env_top4"], f["env_entropy"],
                ))
            med = np.median(np.asarray(rows, dtype=float), axis=0)
            print(
                f"{name:20s}{med[0]:7.3f} {med[1]:7.3f} {med[2]:6.3f}  "
                f"{med[3]:7.3f} {med[4]:7.3f} {med[5]:6.3f}  "
                f"{med[6]:7.3f} {med[7]:7.3f} {med[8]:6.3f}"
            )
        print()

    print("What to look for:")
    print("  * BIPHASE/Barker may concentrate strongly in x^2 because pi reversals vanish.")
    print("  * POLYPHASE may show stronger x^4 structure than x^2.")
    print("  * DUAL_TONE should show strong envelope/cyclic beating in |x|^2.")
    print("  * FM should generally remain spectrally spread under x^2/x^4.")
    print()
    print("If these features separate the families, use them only as extra evidence in the")
    print("operational classifier rather than replacing the existing phase/trajectory tests.")


if __name__ == "__main__":
    main()
