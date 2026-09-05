from config import *

from simulated_source import SimulatedSource
from pulse_detector import PulseDetector
from pdw_extractor import PDWExtractor
from pulse_sequence import PulseSequenceAnalyzer


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
    sequence = analyzer.analyze(pdws)

    print("S2B Experimental ESM")
    print("--------------------")
    print(f"Configured emitters : {len(SIM_EMITTERS)}")
    print(f"Pulses detected     : {len(pulses)}")
    print()

    for item in sequence:
        pdw = item["pdw"]
        pri_s = item["pri_s"]

        if pri_s is None:
            pri_text = "--------"
        else:
            pri_text = f"{pri_s * 1e6:8.3f} us"

        print(f"{pdw}  PRI={pri_text}")


if __name__ == "__main__":
    main()
