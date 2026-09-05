from config import *

from simulated_source import SimulatedSource
from pulse_detector import PulseDetector
from pdw_extractor import PDWExtractor


def main():

    source = SimulatedSource(
        start_delay_s=SIM_START_DELAY_S,
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
        if_frequency_hz=SIM_IF_FREQUENCY_HZ,
        pulse_width_s=SIM_PULSE_WIDTH_S,
        pri_s=SIM_PRI_S,
        num_pulses=SIM_NUM_PULSES,
        amplitude=SIM_AMPLITUDE,
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

    print("S2B Experimental ESM")
    print("--------------------")
    print(f"Pulses detected : {len(pulses)}")
    print()

    for pulse in pulses:
        pdw = extractor.extract(iq, pulse)
        print(pdw)


if __name__ == "__main__":
    main()