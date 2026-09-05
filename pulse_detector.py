import numpy as np


class PulseDetector:
    def __init__(self, threshold, sample_rate_hz, min_pulse_width_s=0.0):
        self.threshold = threshold
        self.sample_rate_hz = sample_rate_hz
        self.min_pulse_width_s = min_pulse_width_s

    def detect(self, iq):
        magnitude = np.abs(iq)

        above = magnitude > self.threshold
        edges = np.diff(above.astype(np.int8))

        starts = np.where(edges == 1)[0] + 1
        stops = np.where(edges == -1)[0] + 1

        if above[0]:
            starts = np.insert(starts, 0, 0)

        if above[-1]:
            stops = np.append(stops, len(iq))

        pulses = []

        min_samples = int(
            round(self.min_pulse_width_s * self.sample_rate_hz)
        )

        for start, stop in zip(starts, stops):

            width_samples = stop - start

            if width_samples < min_samples:
                continue

            pulses.append(
                {
                    "start_sample": int(start),
                    "stop_sample": int(stop),
                }
            )

        return pulses