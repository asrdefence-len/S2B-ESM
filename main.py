from config import *

from simulated_source import SimulatedSource
from pulse_detector import PulseDetector
from pdw_extractor import PDWExtractor
from pulse_sequence import PulseSequenceAnalyzer
from association import FrequencyAssociator, EvidenceAssociator


def print_groups(title, groups, analyzer):
    print(title)
    print("-" * len(title))

    for group in groups:
        sequence = analyzer.analyze(group["pdws"])
        candidate_id = group["candidate_id"]
        mean_frequency_mhz = group["mean_frequency_hz"] / 1e6

        extra = ""
        if "mean_pulse_width_s" in group:
            mean_pw_us = group["mean_pulse_width_s"] * 1e6
            estimated_pri_s = group.get("estimated_pri_s")
            if estimated_pri_s is None:
                pri_text = "unknown"
            else:
                pri_text = f"{estimated_pri_s * 1e6:.3f} us"
            extra = f", mean PW={mean_pw_us:.3f} us, estimated PRI={pri_text}"

        print(
            f"Candidate {candidate_id}: "
            f"{len(group['pdws'])} pulses, "
            f"mean frequency={mean_frequency_mhz:.3f} MHz"
            f"{extra}"
        )

        for item in sequence:
            pdw = item["pdw"]
            pri_s = item["pri_s"]

            if pri_s is None:
                pri_text = "--------"
            else:
                pri_text = f"{pri_s * 1e6:8.3f} us"

            print(
                f"  PDW {pdw.pdw_id:06d}  "
                f"PW={pdw.pulse_width_s * 1e6:6.3f} us  "
                f"PRI={pri_text}"
            )

        print()


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
    pdws = [extractor.extract(iq, pulse) for pulse in pulses]

    analyzer = PulseSequenceAnalyzer()
    raw_sequence = analyzer.analyze(pdws)

    frequency_associator = FrequencyAssociator(
        frequency_tolerance_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ
    )
    frequency_groups = frequency_associator.associate(pdws)

    evidence_associator = EvidenceAssociator(
        frequency_tolerance_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,
        pulse_width_tolerance_s=ASSOCIATION_PULSE_WIDTH_TOLERANCE_S,
        timing_tolerance_s=ASSOCIATION_TIMING_TOLERANCE_S,
    )
    evidence_groups = evidence_associator.associate(pdws)

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
    print_groups(
        "Baseline candidate groups - frequency association only",
        frequency_groups,
        analyzer,
    )

    print_groups(
        "Evidence candidate groups - frequency + PW + PRI consistency",
        evidence_groups,
        analyzer,
    )


if __name__ == "__main__":
    main()
