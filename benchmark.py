import argparse
import math
import time

import numpy as np

from pdw import PDW
from probabilistic_mht import ProbabilisticMultipleHypothesisAssociator
from fast_probabilistic_mht import FastProbabilisticMultipleHypothesisAssociator
from change_detection import PDWChangeDetector


SAMPLE_RATE_HZ = 40_000_000.0
CENTER_FREQUENCY_HZ = 1_000_000_000.0


def make_pdw_stream(count, pulse_rate_hz):
    """Generate a deterministic two-family PDW stream without IQ simulation."""
    interval_s = 1.0 / pulse_rate_hz
    pdws = []
    for index in range(count):
        family = index & 1
        if family == 0:
            frequency_hz = CENTER_FREQUENCY_HZ + 2.000e6
            pulse_width_s = 5.0e-6
            amplitude_dbfs = -6.0
            modulation = "CW"
            bandwidth_hz = 5_000.0
        else:
            frequency_hz = CENTER_FREQUENCY_HZ + 2.080e6
            pulse_width_s = 5.4e-6
            amplitude_dbfs = -8.0
            modulation = "LFM"
            bandwidth_hz = 1.0e6
        pdws.append(PDW(index + 1, index * interval_s, pulse_width_s, frequency_hz,
                        amplitude_dbfs, modulation, bandwidth_hz))
    return pdws


def make_iq_pulses(count, samples_per_pulse=256):
    n = np.arange(samples_per_pulse, dtype=np.float64)
    t = n / SAMPLE_RATE_HZ
    cw = np.exp(1j * 2.0 * np.pi * 2.0e6 * t).astype(np.complex64)
    pulse_duration_s = samples_per_pulse / SAMPLE_RATE_HZ
    chirp_rate = 1.0e6 / pulse_duration_s
    phase = 2.0 * np.pi * (1.5e6 * t + 0.5 * chirp_rate * t * t)
    lfm = np.exp(1j * phase).astype(np.complex64)
    pulses = np.empty((count, samples_per_pulse), dtype=np.complex64)
    pulses[0::2] = cw
    pulses[1::2] = lfm
    return pulses


def modulation_features_batch(pulses, fft_size):
    samples_per_pulse = pulses.shape[1]
    if fft_size < samples_per_pulse:
        work = pulses[:, :fft_size]
    elif fft_size > samples_per_pulse:
        work = np.zeros((pulses.shape[0], fft_size), dtype=np.complex64)
        work[:, :samples_per_pulse] = pulses
    else:
        work = pulses
    window = np.hanning(work.shape[1]).astype(np.float32)
    spectrum = np.fft.fft(work * window[None, :], n=fft_size, axis=1)
    power = np.abs(spectrum) ** 2
    peak_bin = np.argmax(power, axis=1)
    phase_step = np.angle(pulses[:, 1:] * np.conj(pulses[:, :-1]))
    inst_freq_hz = phase_step * (SAMPLE_RATE_HZ / (2.0 * np.pi))
    mean_if = np.mean(inst_freq_hz, axis=1)
    std_if = np.std(inst_freq_hz, axis=1)
    quarter = max(1, inst_freq_hz.shape[1] // 4)
    first = np.mean(inst_freq_hz[:, :quarter], axis=1)
    last = np.mean(inst_freq_hz[:, -quarter:], axis=1)
    duration_s = max((inst_freq_hz.shape[1] - 1) / SAMPLE_RATE_HZ, 1e-12)
    chirp_rate_hz_per_s = (last - first) / duration_s
    return peak_bin, mean_if, std_if, chirp_rate_hz_per_s


def time_call(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start


def rate_text(items, elapsed):
    return "inf" if elapsed <= 0.0 else f"{items / elapsed:,.0f}/s"


def make_mht(cls, args):
    return cls(
        frequency_sigma_hz=250_000.0, pulse_width_sigma_s=1.0e-6,
        amplitude_sigma_db=3.0, timing_sigma_s=80e-6,
        beam_width=args.mht_beam, max_emitters=4,
        birth_probability=0.03, clutter_probability=0.005,
        modulation_match_probability=0.98, missed_pulse_probability=0.08,
        max_pri_multiple=4,
    )


def run_benchmark(args):
    target_pulses = max(1, int(round(args.pulse_rate * args.duration)))
    print("S2B PYTHON THROUGHPUT BENCHMARK\n===============================")
    print(f"Target pulse rate       : {args.pulse_rate:,.0f} pulses/s")
    print(f"Target duration         : {args.duration:.3f} s")
    print(f"Target pulse count      : {target_pulses:,}")
    print(f"Nominal IQ sample rate  : {SAMPLE_RATE_HZ / 1e6:.1f} MS/s")
    print(f"FFT size                : {args.fft_size}\n")

    pdws, elapsed = time_call(make_pdw_stream, target_pulses, args.pulse_rate)
    pdw_rate = target_pulses / elapsed
    print("1. PDW OBJECT GENERATION")
    print(f"   Time                  : {elapsed:.4f} s")
    print(f"   Throughput            : {rate_text(target_pulses, elapsed)}")
    print(f"   Real-time margin      : {pdw_rate / args.pulse_rate:.1f} x target\n")

    feature_batch = max(1, min(args.feature_batch, target_pulses))
    processed = 0
    feature_elapsed = 0.0
    while processed < target_pulses:
        count = min(feature_batch, target_pulses - processed)
        pulses = make_iq_pulses(count, args.pulse_samples)
        _, elapsed = time_call(modulation_features_batch, pulses, args.fft_size)
        feature_elapsed += elapsed
        processed += count
    feature_rate = processed / feature_elapsed
    print("2. INTRA-PULSE FEATURES (FFT + PHASE)")
    print(f"   Pulses processed      : {processed:,}")
    print(f"   Compute time          : {feature_elapsed:.4f} s")
    print(f"   Throughput            : {rate_text(processed, feature_elapsed)}")
    print(f"   Real-time margin      : {feature_rate / args.pulse_rate:.1f} x target\n")

    change_count = min(target_pulses, args.change_pulses)
    _, elapsed = time_call(PDWChangeDetector().detect, pdws[:change_count])
    change_rate = change_count / elapsed if elapsed else math.inf
    print("3. CHANGE-DETECTION PASS")
    print(f"   PDWs tested           : {change_count:,}")
    print(f"   Effective throughput  : {rate_text(change_count, elapsed)}")
    print(f"   Target-rate ratio     : {change_rate / args.pulse_rate:.2f} x\n")

    reference_count = min(target_pulses, args.mht_pulses)
    _, reference_elapsed = time_call(
        make_mht(ProbabilisticMultipleHypothesisAssociator, args).associate,
        pdws[:reference_count],
    )
    reference_rate = reference_count / reference_elapsed
    print("4. REFERENCE PROBABILISTIC MHT")
    print(f"   PDWs tested           : {reference_count:,}")
    print(f"   Time                  : {reference_elapsed:.4f} s")
    print(f"   Effective throughput  : {rate_text(reference_count, reference_elapsed)}")
    print(f"   Target-rate ratio     : {reference_rate / args.pulse_rate:.4f} x\n")

    fast_count = min(target_pulses, args.fast_mht_pulses)
    fast_hypotheses, fast_elapsed = time_call(
        make_mht(FastProbabilisticMultipleHypothesisAssociator, args).associate,
        pdws[:fast_count],
    )
    fast_rate = fast_count / fast_elapsed
    print("5. LIGHTWEIGHT PROBABILISTIC MHT")
    print(f"   PDWs tested           : {fast_count:,}")
    print(f"   Beam width            : {args.mht_beam}")
    print(f"   Time                  : {fast_elapsed:.4f} s")
    print(f"   Effective throughput  : {rate_text(fast_count, fast_elapsed)}")
    print(f"   Target-rate ratio     : {fast_rate / args.pulse_rate:.4f} x")
    print(f"   Retained hypotheses   : {len(fast_hypotheses)}")
    print(f"   Speedup vs reference  : {fast_rate / reference_rate:.1f} x\n")

    print("SUMMARY\n-------")
    print(f"FFT/phase feature path  : {'PASS' if feature_rate >= args.pulse_rate else 'FAIL'}")
    print(f"Reference PMHT          : {'PASS' if reference_rate >= args.pulse_rate else 'FAIL'}")
    print(f"Lightweight PMHT        : {'PASS' if fast_rate >= args.pulse_rate else 'FAIL'}")
    print("\nThe lightweight PMHT is still experimental. First establish throughput, then")
    print("run the functional scenarios before replacing the reference implementation.")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark S2B Python processing throughput")
    parser.add_argument("--pulse-rate", type=float, default=20_000.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fft-size", type=int, default=256)
    parser.add_argument("--pulse-samples", type=int, default=256)
    parser.add_argument("--feature-batch", type=int, default=2_000)
    parser.add_argument("--change-pulses", type=int, default=5_000)
    parser.add_argument("--mht-pulses", type=int, default=250,
                        help="PDWs for slow reference MHT")
    parser.add_argument("--fast-mht-pulses", type=int, default=20_000,
                        help="PDWs for lightweight MHT")
    parser.add_argument("--mht-beam", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
