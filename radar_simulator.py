# radar_simulator.py

import numpy as np

from pulse_timing import generate_pulse_toas


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
        pri_jitter_s=0.0,
        random_seed=0,
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
        self.pri_jitter_s = pri_jitter_s
        self.random_seed = random_seed
        self.pulse_toas_s = generate_pulse_toas(
            start_delay_s=start_delay_s,
            pri_s=pri_s,
            num_pulses=num_pulses,
            pri_jitter_s=pri_jitter_s,
            random_seed=random_seed,
        )

    @property
    def duration_s(self):
        if not self.pulse_toas_s:
            return self.start_delay_s
        return self.pulse_toas_s[-1] + self.pri_s

    def _pulse_waveform(self, num_samples):
        pulse_t = np.arange(num_samples) / self.sample_rate_hz

        if self.modulation == "LFM":
            chirp_rate_hz_per_s = self.lfm_bandwidth_hz / self.pulse_width_s
            start_frequency_hz = self.if_frequency_hz - 0.5 * self.lfm_bandwidth_hz
            phase = 2.0 * np.pi * (
                start_frequency_hz * pulse_t
                + 0.5 * chirp_rate_hz_per_s * pulse_t**2
            )
        else:
            phase = 2.0 * np.pi * self.if_frequency_hz * pulse_t

        return (self.amplitude * np.exp(1j * phase)).astype(np.complex64)

    def generate(self, duration_s=None):
        if duration_s is None:
            duration_s = self.duration_s

        num_samples = int(round(duration_s * self.sample_rate_hz))
        iq = np.zeros(num_samples, dtype=np.complex64)
        pulse_samples = int(round(self.pulse_width_s * self.sample_rate_hz))
        pulse_waveform = self._pulse_waveform(pulse_samples)

        for toa_s in self.pulse_toas_s:
            start = int(round(toa_s * self.sample_rate_hz))
            if start >= num_samples:
                break
            stop = min(start + pulse_samples, num_samples)
            iq[start:stop] += pulse_waveform[: stop - start]

        return iq
