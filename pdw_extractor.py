import numpy as np

from config import MODULATION_LFM_BANDWIDTH_THRESHOLD_HZ
from pdw import PDW


class PDWExtractor:
    def __init__(self, sample_rate_hz, center_frequency_hz):
        self.sample_rate_hz = sample_rate_hz
        self.center_frequency_hz = center_frequency_hz
        self.next_pdw_id = 1

    def _instantaneous_frequency(self, pulse_iq):
        if len(pulse_iq) < 2:
            return np.array([], dtype=float)

        phase_product = pulse_iq[1:] * np.conj(pulse_iq[:-1])
        return (
            np.angle(phase_product)
            * self.sample_rate_hz
            / (2.0 * np.pi)
        )

    def _estimate_frequency_offset(self, pulse_iq):
        instantaneous_frequency = self._instantaneous_frequency(pulse_iq)

        if len(instantaneous_frequency) == 0:
            return 0.0

        return float(np.median(instantaneous_frequency))

    def _estimate_modulation(self, pulse_iq):
        instantaneous_frequency = self._instantaneous_frequency(pulse_iq)

        if len(instantaneous_frequency) < 8:
            return "UNKNOWN", 0.0

        # Trim the noisiest instantaneous-frequency samples at the tails.
        low_hz, high_hz = np.percentile(
            instantaneous_frequency,
            [10.0, 90.0],
        )
        bandwidth_hz = float(max(0.0, high_hz - low_hz))

        if bandwidth_hz >= MODULATION_LFM_BANDWIDTH_THRESHOLD_HZ:
            modulation_type = "LFM"
        else:
            modulation_type = "CW"

        return modulation_type, bandwidth_hz

    def extract(self, iq, pulse):
        start = pulse["start_sample"]
        stop = pulse["stop_sample"]
        pulse_iq = iq[start:stop]

        toa_s = start / self.sample_rate_hz
        pulse_width_s = (stop - start) / self.sample_rate_hz

        amplitude_linear = np.sqrt(np.mean(np.abs(pulse_iq) ** 2))
        amplitude_dbfs = 20.0 * np.log10(max(amplitude_linear, 1e-12))

        frequency_offset_hz = self._estimate_frequency_offset(pulse_iq)
        frequency_hz = self.center_frequency_hz + frequency_offset_hz

        modulation_type, modulation_bandwidth_hz = self._estimate_modulation(
            pulse_iq
        )

        pdw = PDW(
            pdw_id=self.next_pdw_id,
            toa_s=toa_s,
            pulse_width_s=pulse_width_s,
            frequency_hz=frequency_hz,
            amplitude_dbfs=amplitude_dbfs,
            modulation_type=modulation_type,
            modulation_bandwidth_hz=modulation_bandwidth_hz,
        )

        self.next_pdw_id += 1
        return pdw
