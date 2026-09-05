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
        noise_std=0.0,
        start_delay_s=0.0,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.if_frequency_hz = if_frequency_hz
        self.pulse_width_s = pulse_width_s
        self.pri_s = pri_s
        self.num_pulses = num_pulses
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.start_delay_s = start_delay_s

    def generate(self):
        total_time_s = (
            self.start_delay_s
            + self.num_pulses * self.pri_s
        )

        num_samples = int(round(
            total_time_s * self.sample_rate_hz
        ))

        t = np.arange(num_samples) / self.sample_rate_hz

        iq = np.zeros(num_samples, dtype=np.complex64)

        pulse_samples = int(round(
            self.pulse_width_s * self.sample_rate_hz
        ))

        pri_samples = int(round(
            self.pri_s * self.sample_rate_hz
        ))

        start_delay_samples = int(round(
            self.start_delay_s * self.sample_rate_hz
        ))

        for pulse_index in range(self.num_pulses):
            start = (
                start_delay_samples
                + pulse_index * pri_samples
            )

            stop = min(
                start + pulse_samples,
                num_samples
            )

            pulse_t = t[start:stop]

            iq[start:stop] = (
                self.amplitude
                * np.exp(
                    1j
                    * 2
                    * np.pi
                    * self.if_frequency_hz
                    * pulse_t
                )
            )
        if self.noise_std > 0:
            noise = (
                np.random.normal(0, self.noise_std, num_samples)
                + 1j * np.random.normal(
                    0,
                    self.noise_std,
                    num_samples
                )
            )

            iq += noise.astype(np.complex64)

        return iq