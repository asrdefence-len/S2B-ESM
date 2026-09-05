import numpy as np

from radar_simulator import RadarSimulator


class SimulatedSource:
    def __init__(
        self,
        sample_rate_hz,
        center_frequency_hz,
        emitters,
        noise_std,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.center_frequency_hz = center_frequency_hz
        self.noise_std = noise_std

        self.radars = [
            RadarSimulator(
                sample_rate_hz=sample_rate_hz,
                if_frequency_hz=emitter["if_frequency_hz"],
                pulse_width_s=emitter["pulse_width_s"],
                pri_s=emitter["pri_s"],
                num_pulses=emitter["num_pulses"],
                amplitude=emitter["amplitude"],
                start_delay_s=emitter["start_delay_s"],
                modulation=emitter.get("modulation", "CW"),
                lfm_bandwidth_hz=emitter.get("lfm_bandwidth_hz", 0.0),
                pri_jitter_s=emitter.get("pri_jitter_s", 0.0),
                random_seed=emitter.get("random_seed", 0),
            )
            for emitter in emitters
        ]

    def read(self):
        duration_s = max(radar.duration_s for radar in self.radars)
        num_samples = int(round(duration_s * self.sample_rate_hz))
        iq = np.zeros(num_samples, dtype=np.complex64)

        for radar in self.radars:
            iq += radar.generate(duration_s=duration_s)

        if self.noise_std > 0:
            noise = (
                np.random.normal(0, self.noise_std, num_samples)
                + 1j * np.random.normal(0, self.noise_std, num_samples)
            )
            iq += noise.astype(np.complex64)

        metadata = {
            "sample_rate_hz": self.sample_rate_hz,
            "center_frequency_hz": self.center_frequency_hz,
            "start_time_s": 0.0,
        }

        return iq, metadata
