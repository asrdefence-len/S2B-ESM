import argparse
import math
import time

import numpy as np

from pdw import PDW
from probabilistic_mht import ProbabilisticMultipleHypothesisAssociator
from fast_probabilistic_mht import FastProbabilisticMultipleHypothesisAssociator
from gated_fast_probabilistic_mht import GatedFastProbabilisticMultipleHypothesisAssociator
from change_detection import PDWChangeDetector

SAMPLE_RATE_HZ = 40_000_000.0
CENTER_FREQUENCY_HZ = 1_000_000_000.0


def make_pdw_stream(count, pulse_rate_hz):
    interval_s = 1.0 / pulse_rate_hz
    pdws = []
    for index in range(count):
        if index & 1:
            values = (CENTER_FREQUENCY_HZ + 2.080e6, 5.4e-6, -8.0, "LFM", 1.0e6)
        else:
            values = (CENTER_FREQUENCY_HZ + 2.000e6, 5.0e-6, -6.0, "CW", 5_000.0)
        pdws.append(PDW(index + 1, index * interval_s, values[1], values[0], values[2], values[3], values[4]))
    return pdws


def make_iq_pulses(count, samples_per_pulse=256):
    n = np.arange(samples_per_pulse, dtype=np.float64)
    t = n / SAMPLE_RATE_HZ
    cw = np.exp(1j * 2.0 * np.pi * 2.0e6 * t).astype(np.complex64)
    chirp_rate = 1.0e6 / (samples_per_pulse / SAMPLE_RATE_HZ)
    phase = 2.0 * np.pi * (1.5e6 * t + 0.5 * chirp_rate * t * t)
    lfm = np.exp(1j * phase).astype(np.complex64)
    pulses = np.empty((count, samples_per_pulse), dtype=np.complex64)
    pulses[0::2] = cw
    pulses[1::2] = lfm
    return pulses


def modulation_features_batch(pulses, fft_size):
    if fft_size < pulses.shape[1]:
        work = pulses[:, :fft_size]
    elif fft_size > pulses.shape[1]:
        work = np.zeros((pulses.shape[0], fft_size), dtype=np.complex64)
        work[:, :pulses.shape[1]] = pulses
    else:
        work = pulses
    spectrum = np.fft.fft(work * np.hanning(work.shape[1])[None, :], n=fft_size, axis=1)
    peak_bin = np.argmax(np.abs(spectrum) ** 2, axis=1)
    phase_step = np.angle(pulses[:, 1:] * np.conj(pulses[:, :-1]))
    inst_freq = phase_step * (SAMPLE_RATE_HZ / (2.0 * np.pi))
    quarter = max(1, inst_freq.shape[1] // 4)
    duration = max((inst_freq.shape[1] - 1) / SAMPLE_RATE_HZ, 1e-12)
    return (peak_bin, np.mean(inst_freq, axis=1), np.std(inst_freq, axis=1),
            (np.mean(inst_freq[:, -quarter:], axis=1) - np.mean(inst_freq[:, :quarter], axis=1)) / duration)


def time_call(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start


def rate_text(items, elapsed):
    return "inf" if elapsed <= 0.0 else f"{items / elapsed:,.0f}/s"


def make_mht(cls, args):
    return cls(frequency_sigma_hz=250_000.0, pulse_width_sigma_s=1.0e-6,
               amplitude_sigma_db=3.0, timing_sigma_s=80e-6,
               beam_width=args.mht_beam, max_emitters=4,
               birth_probability=0.03, clutter_probability=0.005,
               modulation_match_probability=0.98, missed_pulse_probability=0.08,
               max_pri_multiple=4)


def print_mht_result(number, title, count, elapsed, target_rate, hypotheses, reference_rate):
    rate = count / elapsed
    print(f"{number}. {title}")
    print(f"   PDWs tested           : {count:,}")
    print(f"   Time                  : {elapsed:.4f} s")
    print(f"   Effective throughput  : {rate_text(count, elapsed)}")
    print(f"   Target-rate ratio     : {rate / target_rate:.4f} x")
    print(f"   Retained hypotheses   : {len(hypotheses)}")
    print(f"   Speedup vs reference  : {rate / reference_rate:.1f} x\n")
    return rate


def run_benchmark(args):
    target_pulses = max(1, int(round(args.pulse_rate * args.duration)))
    print("S2B PYTHON THROUGHPUT BENCHMARK\n===============================")
    print(f"Target pulse rate       : {args.pulse_rate:,.0f} pulses/s")
    print(f"Target pulse count      : {target_pulses:,}")
    print(f"Nominal IQ sample rate  : {SAMPLE_RATE_HZ / 1e6:.1f} MS/s")
    print(f"FFT size                : {args.fft_size}\n")

    pdws, elapsed = time_call(make_pdw_stream, target_pulses, args.pulse_rate)
    pdw_rate = target_pulses / elapsed
    print("1. PDW OBJECT GENERATION")
    print(f"   Throughput            : {rate_text(target_pulses, elapsed)}")
    print(f"   Real-time margin      : {pdw_rate / args.pulse_rate:.1f} x target\n")

    processed = 0
    feature_elapsed = 0.0
    while processed < target_pulses:
        count = min(args.feature_batch, target_pulses - processed)
        pulses = make_iq_pulses(count, args.pulse_samples)
        _, elapsed = time_call(modulation_features_batch, pulses, args.fft_size)
        feature_elapsed += elapsed
        processed += count
    feature_rate = processed / feature_elapsed
    print("2. INTRA-PULSE FEATURES (FFT + PHASE)")
    print(f"   Throughput            : {rate_text(processed, feature_elapsed)}")
    print(f"   Real-time margin      : {feature_rate / args.pulse_rate:.1f} x target\n")

    change_count = min(target_pulses, args.change_pulses)
    _, elapsed = time_call(PDWChangeDetector().detect, pdws[:change_count])
    change_rate = change_count / elapsed if elapsed else math.inf
    print("3. CHANGE-DETECTION PASS")
    print(f"   Effective throughput  : {rate_text(change_count, elapsed)}")
    print(f"   Target-rate ratio     : {change_rate / args.pulse_rate:.2f} x\n")

    reference_count = min(target_pulses, args.mht_pulses)
    reference_h, reference_elapsed = time_call(make_mht(ProbabilisticMultipleHypothesisAssociator, args).associate, pdws[:reference_count])
    reference_rate = reference_count / reference_elapsed
    print_mht_result(4, "REFERENCE PROBABILISTIC MHT", reference_count, reference_elapsed, args.pulse_rate, reference_h, reference_rate)

    fast_count = min(target_pulses, args.fast_mht_pulses)
    fast_h, fast_elapsed = time_call(make_mht(FastProbabilisticMultipleHypothesisAssociator, args).associate, pdws[:fast_count])
    fast_rate = print_mht_result(5, "LIGHTWEIGHT PROBABILISTIC MHT", fast_count, fast_elapsed, args.pulse_rate, fast_h, reference_rate)

    gated_count = min(target_pulses, args.gated_mht_pulses)
    gated_h, gated_elapsed = time_call(make_mht(GatedFastProbabilisticMultipleHypothesisAssociator, args).associate, pdws[:gated_count])
    gated_rate = print_mht_result(6, "GATED LIGHTWEIGHT PROBABILISTIC MHT", gated_count, gated_elapsed, args.pulse_rate, gated_h, reference_rate)
    print(f"   Speedup vs lightweight: {gated_rate / fast_rate:.1f} x\n")

    print("SUMMARY\n-------")
    print(f"FFT/phase feature path  : {'PASS' if feature_rate >= args.pulse_rate else 'FAIL'}")
    print(f"Reference PMHT          : {'PASS' if reference_rate >= args.pulse_rate else 'FAIL'}")
    print(f"Lightweight PMHT        : {'PASS' if fast_rate >= args.pulse_rate else 'FAIL'}")
    print(f"Gated lightweight PMHT  : {'PASS' if gated_rate >= args.pulse_rate else 'FAIL'}")
    print("\nGating is deliberately permissive and is a computational pre-filter, not an emitter decision.")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark S2B Python processing throughput")
    parser.add_argument("--pulse-rate", type=float, default=20_000.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fft-size", type=int, default=256)
    parser.add_argument("--pulse-samples", type=int, default=256)
    parser.add_argument("--feature-batch", type=int, default=2_000)
    parser.add_argument("--change-pulses", type=int, default=5_000)
    parser.add_argument("--mht-pulses", type=int, default=250)
    parser.add_argument("--fast-mht-pulses", type=int, default=20_000)
    parser.add_argument("--gated-mht-pulses", type=int, default=20_000)
    parser.add_argument("--mht-beam", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
