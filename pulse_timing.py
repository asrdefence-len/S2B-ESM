import numpy as np


def generate_pulse_toas(
    start_delay_s,
    pri_s,
    num_pulses,
    pri_jitter_s=0.0,
    random_seed=0,
):
    """Generate deterministic pulse TOAs for simulation and truth scoring."""
    if num_pulses <= 0:
        return []

    toas = [float(start_delay_s)]
    if num_pulses == 1:
        return toas

    rng = np.random.default_rng(random_seed)
    for _ in range(1, num_pulses):
        jitter_s = (
            rng.uniform(-pri_jitter_s, pri_jitter_s)
            if pri_jitter_s > 0.0
            else 0.0
        )
        toas.append(toas[-1] + pri_s + jitter_s)

    return toas
