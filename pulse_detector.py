import numpy as np


class PulseDetector:
    def __init__(self, threshold, sample_rate_hz, min_pulse_width_s=0.0,
                 max_internal_gap_s=0.5e-6):
        self.threshold = threshold
        self.sample_rate_hz = sample_rate_hz
        self.min_pulse_width_s = min_pulse_width_s
        self.max_internal_gap_s = max_internal_gap_s

    def _bridge_short_gaps(self, above):
        """Fill brief below-threshold holes inside an otherwise continuous pulse.

        At 40 MS/s a noisy 5--7 us pulse can momentarily fall below threshold and
        otherwise be split into several false PDWs. Only bounded zero-runs with
        above-threshold samples on both sides are filled; true inter-pulse gaps
        are left untouched.
        """
        max_gap_samples = int(round(self.max_internal_gap_s * self.sample_rate_hz))
        if max_gap_samples <= 0 or len(above) < 3:
            return above

        bridged = above.copy()
        below = ~above
        edges = np.diff(below.astype(np.int8))
        starts = list(np.where(edges == 1)[0] + 1)
        stops = list(np.where(edges == -1)[0] + 1)

        if below[0]:
            starts.insert(0, 0)
        if below[-1]:
            stops.append(len(below))

        for start, stop in zip(starts, stops):
            gap = stop - start
            bounded = start > 0 and stop < len(above) and above[start - 1] and above[stop]
            if bounded and gap <= max_gap_samples:
                bridged[start:stop] = True

        return bridged

    def detect(self, iq):
        magnitude = np.abs(iq)

        above = magnitude > self.threshold
        above = self._bridge_short_gaps(above)
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
