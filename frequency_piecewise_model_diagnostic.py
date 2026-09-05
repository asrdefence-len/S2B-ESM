import numpy as np

from physics_waveform_frontend_v6 import PhysicsWaveformFrontendV6
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


def segment_cost_prefix(y):
    y = np.asarray(y, dtype=float)
    s1 = np.concatenate(([0.0], np.cumsum(y)))
    s2 = np.concatenate(([0.0], np.cumsum(y * y)))

    def cost(i, j):
        n = j - i
        if n <= 0:
            return np.inf
        sy = s1[j] - s1[i]
        sy2 = s2[j] - s2[i]
        return max(0.0, sy2 - sy * sy / n)

    return cost


def best_piecewise_constant(y, min_segments=3, max_segments=10, min_len=6):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < min_segments * min_len:
        return np.inf, 0

    cost = segment_cost_prefix(y)
    best_bic = np.inf
    best_k = 0

    for k in range(min_segments, max_segments + 1):
        dp = np.full((k + 1, n + 1), np.inf)
        dp[0, 0] = 0.0
        for seg in range(1, k + 1):
            j_start = seg * min_len
            for j in range(j_start, n + 1):
                i_min = (seg - 1) * min_len
                i_max = j - min_len
                best = np.inf
                for i in range(i_min, i_max + 1):
                    prev = dp[seg - 1, i]
                    if np.isfinite(prev):
                        value = prev + cost(i, j)
                        if value < best:
                            best = value
                dp[seg, j] = best

        sse = dp[k, n]
        if not np.isfinite(sse):
            continue
        variance = max(sse / n, 1e-12)
        # k level means plus k-1 boundaries, approximated as 2k parameters.
        bic = n * np.log(variance) + (2 * k) * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    return float(best_bic), int(best_k)


def smooth_model_bic(y, degree):
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.linspace(-1.0, 1.0, n)
    coeff = np.polyfit(x, y, degree)
    pred = np.polyval(coeff, x)
    sse = float(np.sum((y - pred) ** 2))
    variance = max(sse / n, 1e-12)
    params = degree + 1
    bic = n * np.log(variance) + params * np.log(n)
    return float(bic)


def main():
    classifier = PhysicsWaveformFrontendV6(FS)
    width_us = 7.0
    count = int(round(width_us * 1e-6 * FS))
    names = ("CW", "LFM", "NLFM", "FREQUENCY_CODED")
    snrs = (20, 15, 10)
    seed = 760000

    print("S2B FREQUENCY MODEL-SELECTION DIAGNOSTIC")
    print("========================================")
    print("Compares local-frequency trajectory models using BIC:")
    print("  linear FM, quadratic FM, and K-segment piecewise-constant frequency.")
    print("Lower BIC is better. Delta = piecewise BIC - best smooth-FM BIC.")
    print("Negative Delta strongly favours frequency coding.")
    print()

    for snr in snrs:
        print(f"SNR = {snr} dB")
        print("waveform             BIC-linear   BIC-quad   BIC-piece   K   Delta")
        print("----------------------------------------------------------------")
        for name in names:
            rows = []
            for _ in range(30):
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
                phase = np.unwrap(np.angle(obs))
                local_f = classifier._local_frequency(phase)
                if len(local_f) < 40:
                    continue
                b1 = smooth_model_bic(local_f, 1)
                b2 = smooth_model_bic(local_f, 2)
                bp, k = best_piecewise_constant(local_f)
                delta = bp - min(b1, b2)
                rows.append((b1, b2, bp, k, delta))

            med = np.median(np.asarray(rows, dtype=float), axis=0)
            print(
                f"{name:20s}{med[0]:11.1f}  {med[1]:9.1f}  {med[2]:10.1f}  "
                f"{med[3]:3.0f}  {med[4]:8.1f}"
            )
        print()

    print("Desired signature:")
    print("  LFM/NLFM: best smooth model should beat piecewise constant (Delta > 0).")
    print("  FREQUENCY_CODED: piecewise constant should win strongly (Delta < 0),")
    print("  with a small number of persistent segments close to the code-state count.")


if __name__ == "__main__":
    main()
