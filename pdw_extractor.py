import numpy as np

from pdw import PDW


class PDWExtractor:
    def __init__(self, sample_rate_hz, center_frequency_hz):
        self.sample_rate_hz = sample_rate_hz
        self.center_frequency_hz = center_frequency_hz
        self.next_pdw_id = 1

    def _estimate_frequency_offset(self, pulse_iq):
        if len(pulse_iq) < 2:
            return 0.0

        phase_product = pulse_iq[1:] * np.conj(pulse_iq[:-1])

        mean_phase = np.angle(np.sum(phase_product))

        frequency_offset_hz = (
            mean_phase
            * self.sample_rate_hz
            / (2.0 * np.pi)
        )

        return frequency_offset_hz

    def extract(self, iq, pulse):
        start = pulse["start_sample"]
        stop = pulse["stop_sample"]

        pulse_iq = iq[start:stop]

        toa_s = start / self.sample_rate_hz

        pulse_width_s = (
            stop - start
        ) / self.sample_rate_hz

        amplitude_linear = np.sqrt(
            np.mean(np.abs(pulse_iq) ** 2)
        )

        amplitude_dbfs = 20.0 * np.log10(
            max(amplitude_linear, 1e-12)
        )

        frequency_offset_hz = self._estimate_frequency_offset(
            pulse_iq
        )

        frequency_hz = (
            self.center_frequency_hz
            + frequency_offset_hz
        )

        pdw = PDW(
            pdw_id=self.next_pdw_id,
            toa_s=toa_s,
            pulse_width_s=pulse_width_s,
            frequency_hz=frequency_hz,
            amplitude_dbfs=amplitude_dbfs,
        )

        self.next_pdw_id += 1

        return pdw