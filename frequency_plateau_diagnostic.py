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


def plateau_structure(local_f):
    local_f = np.asarray(local_f, dtype=float)
    if len(local_f) < 8:
        return {
            "plateau_fraction": 0.0,
            "transition_count": 0,
            "longest_plateau": 0,
            "median_plateau": 0.0,
            "level_spread_hz": 0.0,
        }

    excursion = float(np.percentile(local_f, 95) - np.percentile(local_f, 5))
    df = np.diff(local_f)

    # Robust slope scale. Continuous FM has sustained nonzero slope, whereas
    # frequency-coded waveforms have long low-slope runs separated by boundaries.
    threshold = max(0.006 * max(excursion, 1.0), 12_000.0)
    flat = np.abs(df) < threshold

    # Convert flat derivative samples into run lengths in the original local-f array.
    runs = []
    start = None
    for i, is_flat in enumerate(flat):
        if is_flat and start is None:
            start = i
        elif not is_flat and start is not None:
            runs.append(i - start + 1)
            start = None
    if start is not None:
        runs.append(len(flat) - start + 1)

    min_run = max(3, int(round(0.04 * len(local_f))))
    persistent = [r for r in runs if r >= min_run]
    longest = max(persistent) if persistent else 0
    median_run = float(np.median(persistent)) if persistent else 0.0

    # Persistent transitions are regions between persistent plateaus. We do not
    # need exact code boundaries yet; just evidence that multiple plateaus exist.
    transition_count = max(0, len(persistent) - 1)
    plateau_fraction = float(sum(persistent) / len(local_f)) if persistent else 0.0

    # Quantise local frequency coarsely to estimate how many stable levels are occupied.
    level_bin = max(0.04 * max(excursion, 1.0), 100_000.0)
    quantized = np.round(local_f / level_bin).astype(int)
    unique, counts = np.unique(quantized, return_counts=True)
    occupied = counts[counts >= min_run]
    level_count = int(len(occupied))
    level_spread = excursion

    return {
        "plateau_fraction": plateau_fraction,
        "transition_count": transition_count,
        "longest_plateau": longest,
        "median_plateau": median_run,
        "level_count": level_count,
        "level_spread_hz": level_spread,
        "threshold_hz": threshold,
    }


def main():
    classifier = PhysicsWaveformFrontendV6(FS)
    width_us = 7.0
    count = int(round(width_us * 1e-6 * FS))
    names = ("CW", "LFM", "NLFM", "FREQUENCY_CODED")
    snrs = (20, 15, 10)
    seed = 720000

    print("S2B LOCAL-FREQUENCY PLATEAU STRUCTURE DIAGNOSTIC")
    print("================================================")
    print("Uses V6 local phase-slope frequency estimates, but analyses persistent plateaus")
    print("rather than a single plateau-fraction threshold.")
    print()

    for snr in snrs:
        print(f"SNR = {snr} dB")
        print("waveform             plateau  transitions  longest  medianRun  levels  freqExc")
        print("                     fraction count        samples   samples           MHz")
        print("----------------------------------------------------------------------------")

        for name in names:
            rows = []
            for _ in range(40):
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
                s = plateau_structure(local_f)
                rows.append((
                    s["plateau_fraction"],
                    s["transition_count"],
                    s["longest_plateau"],
                    s["median_plateau"],
                    s["level_count"],
                    s["level_spread_hz"],
                ))

            med = np.median(np.asarray(rows, dtype=float), axis=0)
            print(
                f"{name:20s}{med[0]:8.3f}   {med[1]:10.1f}  {med[2]:7.1f}  "
                f"{med[3]:9.1f}  {med[4]:6.1f}  {med[5]/1e6:7.3f}"
            )
        print()

    print("Desired separation:")
    print("  CW: one dominant plateau, little frequency excursion.")
    print("  FM: few/no persistent plateaus; frequency changes continuously.")
    print("  FREQUENCY_CODED: multiple persistent plateaus plus several transitions and large excursion.")
    print()
    print("If frequency-coded shows multiple persistent plateaus while FM does not, use that")
    print("structure as an explicit gate in the next classifier revision rather than retuning V6.")


if __name__ == "__main__":
    main()
