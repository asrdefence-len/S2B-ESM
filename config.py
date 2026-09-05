# config.py

SAMPLE_RATE_HZ = 40_000_000
CENTER_FREQUENCY_HZ = 1_000_000_000
SIM_NOISE_STD = 0.02

DETECTION_THRESHOLD = 0.10
MIN_PULSE_WIDTH_S = 0.5e-6
ASSOCIATION_FREQUENCY_TOLERANCE_HZ = 250_000

SIM_EMITTERS = [
    {
        "name": "Radar A",
        "if_frequency_hz": 2_000_000,
        "pulse_width_s": 5e-6,
        "pri_s": 1e-3,
        "num_pulses": 10,
        "amplitude": 0.5,
        "start_delay_s": 200e-6,
    },
    {
        "name": "Radar B",
        "if_frequency_hz": 2_080_000,
        "pulse_width_s": 8e-6,
        "pri_s": 1.3e-3,
        "num_pulses": 8,
        "amplitude": 0.35,
        "start_delay_s": 500e-6,
    },
]
