from config import *

from simulated_source import SimulatedSource
from pulse_detector import PulseDetector
from pdw_extractor import PDWExtractor
from pulse_sequence import PulseSequenceAnalyzer
from association import FrequencyAssociator


def main():

    source = SimulatedSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
        emitters=SIM_EMITTERS,
        noise_std=SIM_NOISE_STD,
    )

    iq, metadata = source.read()

    detector = PulseDetector(
        threshold=DETECTION_THRESHOLD,
        sample_rate_hz=metadata["sample_rate_hz"],
        min_pulse_width_s=MIN_PULSE_WIDTH_S,
    )

    extractor = PDWExtractor(
        sample_rate_hz=metadata["sample_rate_hz"],
        center_frequency_hz=metadata["center_frequency_hz"],
    )

    pulses = detector.detect(iq)

    pdws = []
    for pulse in pulses:
        pdws.append(extractor.extract(iq, pulse))

    analyzer = PulseSequenceAnalyzer()
    raw_sequence = analyzer.analyze(pdws)

    associator = FrequencyAssociator(
        frequency_tolerance_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ
    )
    candidate_emitters = associator.associate(pdws)

    print("S2B Experimental ESM")
    print("--------------------")
    print(f"Configured emitters : {len(SIM_EMITTERS)}")
    print(f"Pulses detected     : {len(pulses)}")
    print()

    print("Raw interleaved PDW stream")
    print("--------------------------")
    for item in raw_sequence:
        pdw = item["pdw"]
        pri_s = item["pri_s"]

        if pri_s is None:
            pri_text = "--------"
        else:
            pri_text = f"{pri_s * 1e6:8.3f} us"

        print(f"{pdw}  PRI={pri_text}")

    print()
    print("Candidate emitter groups - frequency association only")
    print("-----------------------------------------------------")

    for group in candidate_emitters:
        sequence = analyzer.analyze(group["pdws"])
        candidate_id = group["candidate_id"]
        mean_frequency_mhz = group["mean_frequency_hz"] / 1e6

        print(
            f"Candidate {candidate_id}: "
            f"{len(group['pdws'])} pulses, "
            f"mean frequency={mean_frequency_mhz:.3f} MHz"
        )

        for item in sequence:
            pdw = item["pdw"]
            pri_s = item["pri_s"]

            if pri_s is None:
                pri_text = "--------"
            else:
                pri_text = f"{pri_s * 1e6:8.3f} us"

            print(f"  PDW {pdw.pdw_id:06d}  PRI={pri_text}")

        print()


if __name__ == "__main__":
    main()
