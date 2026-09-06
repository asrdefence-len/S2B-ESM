import numpy as np

from pdw import PDW
from operational_waveform_classifier_cyclic import CyclicOperationalWaveformClassifier


class PDWExtractor:
    def __init__(self, sample_rate_hz, center_frequency_hz):
        self.sample_rate_hz = sample_rate_hz
        self.center_frequency_hz = center_frequency_hz
        self.next_pdw_id = 1
        self.waveform_classifier = CyclicOperationalWaveformClassifier(sample_rate_hz)

    def _fit_phase_model(self, pulse_iq):
        """Fit phase(t) = a*t^2 + b*t + c for RF frequency/BW observables.

        Waveform family classification is deliberately separate and is performed
        by CyclicOperationalWaveformClassifier. The quadratic fit remains useful
        as a cheap centre-frequency and swept-bandwidth measurement.
        """
        if len(pulse_iq) < 8:
            return None

        phase = np.unwrap(np.angle(pulse_iq))
        t = np.arange(len(pulse_iq), dtype=float) / self.sample_rate_hz
        t_centered = t - np.mean(t)
        a, b, c = np.polyfit(t_centered, phase, 2)
        return a, b, c, t_centered

    def _estimate_frequency_and_bandwidth(self, pulse_iq, pulse_width_s):
        fit = self._fit_phase_model(pulse_iq)
        if fit is None:
            return 0.0, 0.0

        a, b, _, _ = fit
        center_frequency_offset_hz = b / (2.0 * np.pi)
        chirp_rate_hz_per_s = a / np.pi
        swept_bandwidth_hz = abs(chirp_rate_hz_per_s) * pulse_width_s
        return float(center_frequency_offset_hz), float(swept_bandwidth_hz)

    def extract(self, iq, pulse):
        start = pulse["start_sample"]
        stop = pulse["stop_sample"]
        pulse_iq = iq[start:stop]

        toa_s = start / self.sample_rate_hz
        pulse_width_s = (stop - start) / self.sample_rate_hz

        amplitude_linear = np.sqrt(np.mean(np.abs(pulse_iq) ** 2))
        amplitude_dbfs = 20.0 * np.log10(max(amplitude_linear, 1e-12))

        frequency_offset_hz, modulation_bandwidth_hz = self._estimate_frequency_and_bandwidth(
            pulse_iq, pulse_width_s
        )
        waveform = self.waveform_classifier.classify(pulse_iq)
        frequency_hz = self.center_frequency_hz + frequency_offset_hz

        pdw = PDW(
            pdw_id=self.next_pdw_id,
            toa_s=toa_s,
            pulse_width_s=pulse_width_s,
            frequency_hz=frequency_hz,
            amplitude_dbfs=amplitude_dbfs,
            modulation_type=waveform.family,
            modulation_bandwidth_hz=modulation_bandwidth_hz,
            modulation_confidence=waveform.confidence,
            modulation_scores=dict(waveform.scores),
            modulation_rejection_reason=waveform.rejection_reason,
        )

        self.next_pdw_id += 1
        return pdw
