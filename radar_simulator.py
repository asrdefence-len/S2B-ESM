# radar_simulator.py

import numpy as np


class RadarSimulator:
    def __init__(
        self,
        sample_rate_hz,
        if_frequency_hz,
        pulse_width_s,
        pri_s,
        num_pulses,
        amplitude=0.5,
        start_delay_s=0.0,
        modulation="CW",
        lfm_bandwidth_hz=0.0,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.if_frequency_hz = if_frequency_hz
        self.pulse_width_s = pulse_width_s
        self.pri_s = pri_s
        self.num_pulses = num_pulses
        self.amplitude = amplitude
        self.start_delay_s = start_delay_s
        self.modulation = modulation.upper()
        self.lfm_bandwidth_hz = lfm_bandwidth_hz

    @property
    def duration_s(self):
        return self.start_delay_s + self.num_pulses * self.pri_s

    def _pulse_waveform(self, num_samples):
        pulse_t = np.arange(num_samples) / self.sample_rate_hz

        if self.modulation == "LFM":
            # Symmetric linear chirp around the configured IF frequency.
            chirp_rate_hz_per_s = self.lfm_bandwidth_hz / self.pulse_width_s
            start_frequency_hz = (
                self.if_frequency_hz - 0.5 * self.lfm_bandwidth_hz
            )
            phase = 2.0 * np.pi * (
                start_frequency_hz * pulse_t
                + 0.5 * chirp_rate_hz_per_s * pulse_t**2
            )
        else:
            phase = 2.0 * np.pi * self.if_frequency_hz * pulse_t

        return (
            self.amplitude * np.exp(1j * phase)
        ).astype(np.complex64)

    def generate(self, duration_s=None):
        if duration_s is None:
            duration_s = self.duration_s

        num_samples = int(round(duration_s * self.sample_rate_hz))
        iq = np.zeros(num_samples, dtype=np.complex64)

        pulse_samples = int(round(self.pulse_width_s * self.sample_rate_hz))
        pri_samples = int(round(self.pri_s * self.sample_rate_hz))
        start_delay_samples = int(round(self.start_delay_s * self.sample_rate_hz))
        pulse_waveform = self._pulse_waveform(pulse_samples)

        for pulse_index in range(self.num_pulses):
            start = start_delay_samples + pulse_index * pri_samples

            if start >= num_samples:
                break

            stop = min(start + pulse_samples, num_samples)
            iq[start:stop] += pulse_waveform[: stop - start]

        return iq
