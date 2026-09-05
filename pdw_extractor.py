import numpy as np

from config import MODULATION_LFM_BANDWIDTH_THRESHOLD_HZ
from pdw import PDW


class PDWExtractor:
    def __init__(self, sample_rate_hz, center_frequency_hz):
        self.sample_rate_hz = sample_rate_hz
        self.center_frequency_hz = center_frequency_hz
        self.next_pdw_id = 1

    def _fit_phase_model(self, pulse_iq):
        """Fit phase(t) = a*t^2 + b*t + c to the pulse.

        A CW pulse should have a near-zero quadratic term. An LFM pulse has a
        clear quadratic phase term whose derivative gives a linearly changing
        instantaneous frequency.
        """
        if len(pulse_iq) < 8:
            return None

        phase = np.unwrap(np.angle(pulse_iq))
        t = np.arange(len(pulse_iq), dtype=float) / self.sample_rate_hz
        t_centered = t - np.mean(t)

        a, b, c = np.polyfit(t_centered, phase, 2)
        return a, b, c, t_centered

    def _estimate_frequency_and_modulation(self, pulse_iq, pulse_width_s):
        fit = self._fit_phase_model(pulse_iq)

        if fit is None:
            return 0.0, "UNKNOWN", 0.0

        a, b, _, _ = fit

        # With centred time, b is the phase slope at the pulse midpoint.
        center_frequency_offset_hz = b / (2.0 * np.pi)

        # phase = 2*pi*(f0*t + 0.5*k*t^2), so a = pi*k.
        chirp_rate_hz_per_s = a / np.pi
        swept_bandwidth_hz = abs(chirp_rate_hz_per_s) * pulse_width_s

        modulation_type = (
            "LFM"
            if swept_bandwidth_hz >= MODULATION_LFM_BANDWIDTH_THRESHOLD_HZ
            else "CW"
        )

        return (
            float(center_frequency_offset_hz),
            modulation_type,
            float(swept_bandwidth_hz),
        )

    def extract(self, iq, pulse):
        start = pulse["start_sample"]
        stop = pulse["stop_sample"]
        pulse_iq = iq[start:stop]

        toa_s = start / self.sample_rate_hz
        pulse_width_s = (stop - start) / self.sample_rate_hz

        amplitude_linear = np.sqrt(np.mean(np.abs(pulse_iq) ** 2))
        amplitude_dbfs = 20.0 * np.log10(max(amplitude_linear, 1e-12))

        (
            frequency_offset_hz,
            modulation_type,
            modulation_bandwidth_hz,
        ) = self._estimate_frequency_and_modulation(
            pulse_iq,
            pulse_width_s,
        )

        frequency_hz = self.center_frequency_hz + frequency_offset_hz

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
