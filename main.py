from config import *

from radar_simulator import RadarSimulator
from pulse_detector import PulseDetector
from pdw_extractor import PDWExtractor


def main():

    radar = RadarSimulator(
        sample_rate_hz=SAMPLE_RATE_HZ,
        if_frequency_hz=SIM_IF_FREQUENCY_HZ,
        pulse_width_s=SIM_PULSE_WIDTH_S,
        pri_s=SIM_PRI_S,
        num_pulses=SIM_NUM_PULSES,
        amplitude=SIM_AMPLITUDE,
        noise_std=SIM_NOISE_STD,
    )

    iq = radar.generate()

    detector = PulseDetector(
        threshold=DETECTION_THRESHOLD,
        sample_rate_hz=SAMPLE_RATE_HZ,
        min_pulse_width_s=MIN_PULSE_WIDTH_S,
    )

    extractor = PDWExtractor(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
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