import numpy as np

from physics_waveform_frontend_v4_test import nlfm, stepped_frequency
from variable_bandwidth_waveform_test import clean_waveform
from waveform_classifier_stress_test import impair


FS = 40_000_000.0


def make_waveform(name, count):
    if name == "CW":
        return clean_waveform("CW", count, 0.30)
    if name == "LFM":
        return clean_waveform("LFM", count, 0.25)
    if name == "NLFM":
        return nlfm(count, 0.28)
    if name == "FREQUENCY_CODED":
        return stepped_frequency(count, steps=7, normalized_step=0.045)
    raise ValueError(name)


def phase_metrics(samples):
    x = np.asarray(samples, dtype=np.complex128)
    phase = np.unwrap(np.angle(x))
    n = np.linspace(-1.0, 1.0, len(phase))

    metrics = {}
    for degree, name in ((1, "linear"), (2, "quadratic"), (3, "cubic")):
        coeff = np.polyfit(n, phase, degree)
        pred = np.polyval(coeff, n)
        resid = phase - pred
        metrics[name] = float(np.sqrt(np.mean(resid ** 2)))

    # Local slope estimate from a moving linear regression over unwrapped phase.
    window = max(9, int(round(len(phase) * 0.08)))
    if window % 2 == 0:
        window += 1
    half = window // 2
    slopes = []
    centers = []
    for i in range(half, len(phase) - half):
        idx = np.arange(i - half, i + half + 1, dtype=float)
        y = phase[i - half:i + half + 1]
        a = np.column_stack((np.ones_like(idx), idx))
        coeff, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
        slopes.append(coeff[1] * FS / (2.0 * np.pi))
        centers.append(i)
    slopes = np.asarray(slopes)

    if len(slopes) > 2:
        ds = np.diff(slopes)
        excursion = float(np.percentile(slopes, 95) - np.percentile(slopes, 5))
        median_abs_ds = float(np.median(np.abs(ds)))
        med = float(np.median(ds))
        mad = float(np.median(np.abs(ds - med)))
        scale = max(1.4826 * mad, 1.0)
        jump_threshold = max(6.0 * scale, 50_000.0)
        jumps = int(np.count_nonzero(np.abs(ds - med) > jump_threshold))
    else:
        excursion = 0.0
        median_abs_ds = 0.0
        jumps = 0

    return metrics, excursion, median_abs_ds, jumps


def main():
    width_us = 7.0
    count = int(round(width_us * 1e-6 * FS))
    names = ("CW", "LFM", "NLFM", "FREQUENCY_CODED")
    snrs = (20, 15, 10)

    print("S2B PHASE-POLYNOMIAL TRAJECTORY DIAGNOSTIC")
    print("==========================================")
    print("Fits polynomial models directly to unwrapped pulse phase, then estimates")
    print("instantaneous frequency using local phase-slope regression.")
    print()

    seed = 610000
    for snr in snrs:
        print(f"SNR = {snr} dB")
        print("waveform             phaseRMS1  phaseRMS2  phaseRMS3  freqExc   |dSlope|  jumps")
        print("                     rad        rad        rad        MHz       kHz       count")
        print("----------------------------------------------------------------------------")
        for name in names:
            values = []
            for trial in range(30):
                seed += 1
                rng = np.random.default_rng(seed)
                clean = make_waveform(name, count)
                obs = impair(
                    clean,
                    snr_db=snr,
                    frequency_offset_hz=rng.uniform(-5e6, 5e6),
                    phase_rad=rng.uniform(-np.pi, np.pi),
                    timing_shift=int(rng.integers(-3, 4)),
                    seed=seed,
                )
                m, excursion, ds, jumps = phase_metrics(obs)
                values.append((m["linear"], m["quadratic"], m["cubic"], excursion, ds, jumps))
            med = np.median(np.asarray(values, dtype=float), axis=0)
            print(
                f"{name:20s}{med[0]:9.3f}  {med[1]:9.3f}  {med[2]:9.3f}  "
                f"{med[3]/1e6:8.3f}  {med[4]/1e3:8.3f}  {med[5]:5.1f}"
            )
        print()

    print("Desired signatures:")
    print("  CW: degree-1 phase fit good.")
    print("  LFM: degree-2 phase fit much better than degree-1.")
    print("  NLFM: degree-3 phase fit much better than degree-2.")
    print("  FREQUENCY_CODED: polynomial phase fits remain poor; local slopes show plateaus/jumps.")


if __name__ == "__main__":
    main()
