from radar_simulator import RadarSimulator


class SimulatedSource:
    def __init__(
        self,
        start_delay_s,
        sample_rate_hz,
        center_frequency_hz,
        if_frequency_hz,
        pulse_width_s,
        pri_s,
        num_pulses,
        amplitude,
        noise_std,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.center_frequency_hz = center_frequency_hz

        self.radar = RadarSimulator(
            sample_rate_hz=sample_rate_hz,
            if_frequency_hz=if_frequency_hz,
            pulse_width_s=pulse_width_s,
            pri_s=pri_s,
            num_pulses=num_pulses,
            amplitude=amplitude,
            noise_std=noise_std,
            start_delay_s=start_delay_s
        )

    def read(self):
        iq = self.radar.generate()

        metadata = {
            "sample_rate_hz": self.sample_rate_hz,
            "center_frequency_hz": self.center_frequency_hz,
            "start_time_s": 0.0,
        }

        return iq, metadata