import numpy as np

from physics_waveform_frontend_v4_test import make_test_waveform


FS = 40_000_000.0


def trajectory_metrics(samples):
    x = np.asarray(samples, dtype=np.complex128)
    q = np.unwrap(np.angle(x[1:] * np.conj(x[:-1])))
    f = FS * q / (2.0 * np.pi)
    n = np.linspace(-1.0, 1.0, len(f))

    def fit(degree):
        coeff = np.polyfit(n, f, degree)
        predicted = np.polyval(coeff, n)
        residual = f - predicted
        return float(np.sqrt(np.mean(residual ** 2)))

    constant_rms = fit(0)
    linear_rms = fit(1)
    quadratic_rms = fit(2)

    df = np.diff(f)
    med_df = float(np.median(df)) if len(df) else 0.0
    mad_df = float(np.median(np.abs(df - med_df))) if len(df) else 0.0
    scale = max(1.4826 * mad_df, 1.0)
    jump_threshold = max(6.0 * scale, 50_000.0)
    jumps = int(np.count_nonzero(np.abs(df - med_df) > jump_threshold))

    excursion = float(np.percentile(f, 95) - np.percentile(f, 5))
    smoothness = float(np.median(np.abs(df))) if len(df) else 0.0

    return {
        "excursion": excursion,
        "constant_rms": constant_rms,
        "linear_rms": linear_rms,
        "quadratic_rms": quadratic_rms,
        "jumps": jumps,
        "smoothness": smoothness,
    }


def main():
    rng = np.random.default_rng(440001)
    width_us = 7.0
    count = int(round(width_us * 1e-6 * FS))
    names = ("CW", "LFM", "NLFM", "FREQUENCY_CODED")

    print("S2B INSTANTANEOUS-FREQUENCY TRAJECTORY DIAGNOSTIC")
    print("=================================================")
    print(f"Clean {width_us:.1f} us pulses at {FS/1e6:.1f} MS/s; no noise or carrier offset.")
    print()
    print("waveform             excursion   const RMS  linear RMS   quad RMS   jumps   median |df|")
    print("                     MHz         MHz        MHz          MHz                 kHz/sample")
    print("--------------------------------------------------------------------------------------")

    for name in names:
        x = make_test_waveform(name, count, rng)
        m = trajectory_metrics(x)
        print(
            f"{name:20s}"
            f"{m['excursion']/1e6:9.3f}   "
            f"{m['constant_rms']/1e6:9.3f}  "
            f"{m['linear_rms']/1e6:9.3f}   "
            f"{m['quadratic_rms']/1e6:9.3f}   "
            f"{m['jumps']:5d}   "
            f"{m['smoothness']/1e3:10.3f}"
        )

    print()
    print("Interpretation:")
    print("  CW: very small excursion and constant-model residual.")
    print("  LFM: large excursion, but linear residual should collapse.")
    print("  NLFM: large smooth excursion; quadratic residual should improve materially over linear.")
    print("  FREQUENCY_CODED: long flat regions plus a small number of abrupt frequency jumps.")
    print()
    print("If these clean metrics do not separate the families, fix the waveform generator/features before V5.")


if __name__ == "__main__":
    main()
