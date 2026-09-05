import argparse
import math
import time

import numpy as np

from pdw import PDW
from probabilistic_mht import ProbabilisticMultipleHypothesisAssociator
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

        pdws.append(
            PDW(
                pdw_id=index + 1,
                toa_s=index * interval_s,
                pulse_width_s=pulse_width_s,
                frequency_hz=frequency_hz,
                amplitude_dbfs=amplitude_dbfs,
                modulation_type=modulation,
                modulation_bandwidth_hz=bandwidth_hz,
            )
        )

    return pdws


def make_iq_pulses(count, samples_per_pulse=256):
    """Create alternating CW and LFM complex pulse windows for feature timing."""
    n = np.arange(samples_per_pulse, dtype=np.float64)
    t = n / SAMPLE_RATE_HZ

    cw_frequency_hz = 2.0e6
    cw = np.exp(1j * 2.0 * np.pi * cw_frequency_hz * t).astype(np.complex64)

    lfm_start_hz = 1.5e6
    lfm_bandwidth_hz = 1.0e6
    pulse_duration_s = samples_per_pulse / SAMPLE_RATE_HZ
    chirp_rate_hz_per_s = lfm_bandwidth_hz / pulse_duration_s
    phase = 2.0 * np.pi * (
        lfm_start_hz * t + 0.5 * chirp_rate_hz_per_s * t * t
    )
    lfm = np.exp(1j * phase).astype(np.complex64)

    pulses = np.empty((count, samples_per_pulse), dtype=np.complex64)
    pulses[0::2] = cw
    pulses[1::2] = lfm
    return pulses


def modulation_features_batch(pulses, fft_size):
    """Representative fast per-pulse FFT + phase-difference feature extraction."""
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

    # Phase-difference instantaneous-frequency statistics.
    phase_step = np.angle(pulses[:, 1:] * np.conj(pulses[:, :-1]))
    inst_freq_hz = phase_step * (SAMPLE_RATE_HZ / (2.0 * np.pi))
    mean_if = np.mean(inst_freq_hz, axis=1)
    std_if = np.std(inst_freq_hz, axis=1)

    # Simple chirp slope from first and last quarter means. This deliberately
    # avoids a Python loop and is representative of a cheap real-time feature.
    quarter = max(1, inst_freq_hz.shape[1] // 4)
    first = np.mean(inst_freq_hz[:, :quarter], axis=1)
    last = np.mean(inst_freq_hz[:, -quarter:], axis=1)
    duration_s = max((inst_freq_hz.shape[1] - 1) / SAMPLE_RATE_HZ, 1e-12)
    chirp_rate_hz_per_s = (last - first) / duration_s

    return peak_bin, mean_if, std_if, chirp_rate_hz_per_s


def time_call(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def rate_text(items, elapsed):
    if elapsed <= 0.0:
        return "inf"
    return f"{items / elapsed:,.0f}/s"


def run_benchmark(args):
    target_pulses = max(1, int(round(args.pulse_rate * args.duration)))

    print("S2B PYTHON THROUGHPUT BENCHMARK")
    print("===============================")
    print(f"Target pulse rate       : {args.pulse_rate:,.0f} pulses/s")
    print(f"Target duration         : {args.duration:.3f} s")
    print(f"Target pulse count      : {target_pulses:,}")
    print(f"Nominal IQ sample rate  : {SAMPLE_RATE_HZ / 1e6:.1f} MS/s")
    print(f"FFT size                : {args.fft_size}")
    print()

    pdws, elapsed = time_call(make_pdw_stream, target_pulses, args.pulse_rate)
    pdw_rate = target_pulses / elapsed
    print("1. PDW OBJECT GENERATION")
    print(f"   Time                  : {elapsed:.4f} s")
    print(f"   Throughput            : {rate_text(target_pulses, elapsed)}")
    print(f"   Real-time margin      : {pdw_rate / args.pulse_rate:.1f} x target")
    print()

    # Avoid allocating an arbitrarily huge IQ matrix. Process in batches while
    # still measuring the complete requested pulse count.
    feature_batch = max(1, min(args.feature_batch, target_pulses))
    feature_processed = 0
    feature_elapsed = 0.0

    while feature_processed < target_pulses:
        batch_count = min(feature_batch, target_pulses - feature_processed)
        pulses = make_iq_pulses(batch_count, args.pulse_samples)
        _, elapsed = time_call(modulation_features_batch, pulses, args.fft_size)
        feature_elapsed += elapsed
        feature_processed += batch_count

    feature_rate = feature_processed / feature_elapsed
    print("2. INTRA-PULSE FEATURES (FFT + PHASE)")
    print(f"   Pulse samples         : {args.pulse_samples}")
    print(f"   Pulses processed      : {feature_processed:,}")
    print(f"   Compute time          : {feature_elapsed:.4f} s")
    print(f"   Throughput            : {rate_text(feature_processed, feature_elapsed)}")
    print(f"   Real-time margin      : {feature_rate / args.pulse_rate:.1f} x target")
    print()

    change_count = min(target_pulses, args.change_pulses)
    change_pdws = pdws[:change_count]
    change_detector = PDWChangeDetector()
    _, elapsed = time_call(change_detector.detect, change_pdws)
    change_rate = change_count / elapsed if elapsed > 0.0 else math.inf
    print("3. CHANGE-DETECTION PASS")
    print(f"   PDWs tested           : {change_count:,}")
    print(f"   Time                  : {elapsed:.4f} s")
    print(f"   Effective throughput  : {rate_text(change_count, elapsed)}")
    print(f"   Target-rate ratio     : {change_rate / args.pulse_rate:.2f} x")
    print("   Note                  : current detector is a batch diagnostic, not yet streaming")
    print()

    mht_count = min(target_pulses, args.mht_pulses)
    mht_pdws = pdws[:mht_count]
    mht = ProbabilisticMultipleHypothesisAssociator(
        frequency_sigma_hz=250_000.0,
        pulse_width_sigma_s=1.0e-6,
        amplitude_sigma_db=3.0,
        timing_sigma_s=80e-6,
        beam_width=args.mht_beam,
        max_emitters=4,
        birth_probability=0.03,
        clutter_probability=0.005,
        modulation_match_probability=0.98,
        missed_pulse_probability=0.08,
        max_pri_multiple=4,
    )
    hypotheses, elapsed = time_call(mht.associate, mht_pdws)
    mht_rate = mht_count / elapsed if elapsed > 0.0 else math.inf
    print("4. CURRENT PROBABILISTIC MHT")
    print(f"   PDWs tested           : {mht_count:,}")
    print(f"   Beam width            : {args.mht_beam}")
    print(f"   Time                  : {elapsed:.4f} s")
    print(f"   Effective throughput  : {rate_text(mht_count, elapsed)}")
    print(f"   Target-rate ratio     : {mht_rate / args.pulse_rate:.4f} x")
    print(f"   Retained hypotheses   : {len(hypotheses)}")
    print("   Note                  : uses deepcopy and is expected to be the present bottleneck")
    print()

    print("SUMMARY")
    print("-------")
    print(
        f"FFT/phase feature path  : {'PASS' if feature_rate >= args.pulse_rate else 'FAIL'} "
        f"for {args.pulse_rate:,.0f} pulses/s target"
    )
    print(
        f"Current PMHT path       : {'PASS' if mht_rate >= args.pulse_rate else 'FAIL'} "
        f"for {args.pulse_rate:,.0f} pulses/s target"
    )
    print()
    print("This benchmark intentionally measures stages separately. It does not claim")
    print("end-to-end real-time operation until acquisition, queues and streaming track")
    print("management are implemented and measured together.")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark S2B Python processing throughput")
    parser.add_argument("--pulse-rate", type=float, default=20_000.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fft-size", type=int, default=256)
    parser.add_argument("--pulse-samples", type=int, default=256)
    parser.add_argument("--feature-batch", type=int, default=2_000)
    parser.add_argument("--change-pulses", type=int, default=5_000)
    parser.add_argument("--mht-pulses", type=int, default=250)
    parser.add_argument("--mht-beam", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
