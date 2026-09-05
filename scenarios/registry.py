from .scenario import Scenario


def _emitter(
    name,
    if_frequency_hz,
    pulse_width_s,
    pri_s,
    num_pulses,
    amplitude,
    start_delay_s,
    modulation="CW",
    lfm_bandwidth_hz=0.0,
    pri_jitter_s=0.0,
    random_seed=0,
    physical_emitter_id=None,
    mode="UNSPECIFIED",
):
    return {
        "name": name,
        "physical_emitter_id": physical_emitter_id or name,
        "mode": mode,
        "if_frequency_hz": if_frequency_hz,
        "pulse_width_s": pulse_width_s,
        "pri_s": pri_s,
        "num_pulses": num_pulses,
        "amplitude": amplitude,
        "start_delay_s": start_delay_s,
        "modulation": modulation,
        "lfm_bandwidth_hz": lfm_bandwidth_hz,
        "pri_jitter_s": pri_jitter_s,
        "random_seed": random_seed,
    }


SCENARIOS = {
    "one_emitter": Scenario(
        name="one_emitter",
        description="Single stable CW radar. Basic pulse detection and PDW sanity check.",
        noise_std=0.02,
        emitters=[
            _emitter("Radar A", 2_000_000, 5.0e-6, 1.0e-3, 10, 0.50, 200e-6),
        ],
    ),
    "two_easy_emitters": Scenario(
        name="two_easy_emitters",
        description="Two well-separated emitters for a straightforward association check.",
        noise_std=0.02,
        emitters=[
            _emitter("Radar A", 1_000_000, 4.0e-6, 1.0e-3, 10, 0.50, 200e-6),
            _emitter(
                "Radar B", 6_000_000, 7.0e-6, 1.5e-3, 7, 0.35, 500e-6,
                modulation="LFM", lfm_bandwidth_hz=1_000_000,
            ),
        ],
    ),
    "close_emitters": Scenario(
        name="close_emitters",
        description="Current S2B baseline: close RFs but different waveform and timing evidence.",
        noise_std=0.02,
        emitters=[
            _emitter("Radar A", 2_000_000, 5.0e-6, 1.0e-3, 10, 0.50, 200e-6),
            _emitter(
                "Radar B", 2_080_000, 5.4e-6, 1.3e-3, 8, 0.40, 500e-6,
                modulation="LFM", lfm_bandwidth_hz=1_000_000,
            ),
        ],
    ),
    "same_modulation": Scenario(
        name="same_modulation",
        description="Two close CW emitters; association must rely on timing and scalar PDW evidence.",
        noise_std=0.02,
        emitters=[
            _emitter("Radar A", 2_000_000, 5.0e-6, 1.0e-3, 10, 0.50, 200e-6),
            _emitter("Radar B", 2_080_000, 5.4e-6, 1.3e-3, 8, 0.40, 500e-6),
        ],
    ),
    "jittered_pri": Scenario(
        name="jittered_pri",
        description="Two close emitters with modest PRI jitter: A +/-20 us, B +/-30 us.",
        noise_std=0.02,
        emitters=[
            _emitter(
                "Radar A", 2_000_000, 5.0e-6, 1.0e-3, 10, 0.50, 200e-6,
                pri_jitter_s=20e-6, random_seed=101,
            ),
            _emitter(
                "Radar B", 2_080_000, 5.4e-6, 1.3e-3, 8, 0.40, 500e-6,
                modulation="LFM", lfm_bandwidth_hz=1_000_000,
                pri_jitter_s=30e-6, random_seed=202,
            ),
        ],
    ),
    "search_to_track": Scenario(
        name="search_to_track",
        description=(
            "One physical radar changes from SEARCH to TRACK. The ESM is not told "
            "that the two waveform segments come from the same physical emitter."
        ),
        noise_std=0.02,
        emitters=[
            _emitter(
                "Radar A SEARCH",
                2_000_000,
                5.0e-6,
                1.0e-3,
                8,
                0.50,
                200e-6,
                modulation="CW",
                physical_emitter_id="Radar A",
                mode="SEARCH",
            ),
            _emitter(
                "Radar A TRACK",
                2_050_000,
                7.0e-6,
                300e-6,
                14,
                0.48,
                7.5e-3,
                modulation="LFM",
                lfm_bandwidth_hz=1_000_000,
                physical_emitter_id="Radar A",
                mode="TRACK",
            ),
        ],
    ),
}


def list_scenarios():
    return list(SCENARIOS.values())


def get_scenario(name):
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(SCENARIOS)
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}") from exc
