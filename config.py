# config.py

SAMPLE_RATE_HZ = 40_000_000
CENTER_FREQUENCY_HZ = 1_000_000_000
SIM_NOISE_STD = 0.02

DETECTION_THRESHOLD = 0.10
MIN_PULSE_WIDTH_S = 0.5e-6

# Association tolerances. These are deliberately broad enough that no single
# scalar parameter is guaranteed to separate the two emitters.
ASSOCIATION_FREQUENCY_TOLERANCE_HZ = 250_000
ASSOCIATION_PULSE_WIDTH_TOLERANCE_S = 1.0e-6
ASSOCIATION_TIMING_TOLERANCE_S = 80e-6
ASSOCIATION_AMPLITUDE_TOLERANCE_DB = 3.0

MODULATION_LFM_BANDWIDTH_THRESHOLD_HZ = 200_000

SIM_EMITTERS = [
    {
        "name": "Radar A",
        "if_frequency_hz": 2_000_000,
        "pulse_width_s": 5.0e-6,
        "pri_s": 1.0e-3,
        "num_pulses": 10,
        "amplitude": 0.50,
        "start_delay_s": 200e-6,
        "modulation": "CW",
        "lfm_bandwidth_hz": 0.0,
    },
    {
        "name": "Radar B",
        "if_frequency_hz": 2_080_000,
        "pulse_width_s": 5.4e-6,
        "pri_s": 1.3e-3,
        "num_pulses": 8,
        "amplitude": 0.40,
        "start_delay_s": 500e-6,
        "modulation": "LFM",
        "lfm_bandwidth_hz": 1_000_000,
    },
]
