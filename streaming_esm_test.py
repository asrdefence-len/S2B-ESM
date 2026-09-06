"""Exercise E1, E2 and E3 through one 40 MS/s sampled-IQ ESM chain.

This is the architectural bridge to an SDR source.  It intentionally stops at
PDWs so the streaming front end can be verified before persistent association
and the operator UI are moved onto it.
"""

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from pdw_extractor import PDWExtractor
from pulse_detector import PulseDetector
from simulated_streaming_source import SimulatedStreamingIQSource


CENTER_HZ = 9_400_000_000
BLOCK_SAMPLES = 40_000       # 1.0 ms at 40 MS/s
RUN_SECONDS = 6.0


def main():
    source = SimulatedStreamingIQSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_HZ,
        block_samples=BLOCK_SAMPLES,
        noise_std=0.02,
    )
    detector = PulseDetector(DETECTION_THRESHOLD, SAMPLE_RATE_HZ, MIN_PULSE_WIDTH_S)
    extractor = PDWExtractor(SAMPLE_RATE_HZ, CENTER_HZ)

    blocks = int(round(RUN_SECONDS * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    pdws = []

    print("S2B UNIFIED 40 MS/s STREAMING FRONT END")
    print("========================================")
    print(f"Sample rate : {SAMPLE_RATE_HZ/1e6:.1f} MS/s")
    print(f"Block size  : {BLOCK_SAMPLES} complex samples ({1000*BLOCK_SAMPLES/SAMPLE_RATE_HZ:.3f} ms)")
    print("Source      : E1 + E2 + scripted E3 mixed into the SAME IQ blocks")
    print("Chain       : IQ -> pulse detector -> waveform classifier -> PDW")
    print("No E3 observation-level bypass is used in this test.\n")

    last_report_s = -1
    for _ in range(blocks):
        iq, metadata = source.read_block()
        pulses = detector.detect(iq)
        for pulse in pulses:
            pdw = extractor.extract(iq, pulse, block_start_time_s=metadata["start_time_s"])
            pdws.append(pdw)

        second = int(metadata["start_time_s"])
        if second != last_report_s:
            recent = [p for p in pdws if p.toa_s >= max(0.0, second - 1.0)]
            print(f"t={metadata['start_time_s']:5.2f}s  total PDWs={len(pdws):6d}  recent={len(recent):5d}")
            last_report_s = second

    print("\nLast 20 measured PDWs:")
    for pdw in pdws[-20:]:
        print(pdw)

    print(f"\nTotal measured PDWs: {len(pdws)}")
    print("PASS: all displayed PDWs above came from sampled 40 MS/s IQ and the common detector/extractor path.")


if __name__ == "__main__":
    main()
