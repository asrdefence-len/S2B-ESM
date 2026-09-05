import numpy as np

from physics_waveform_frontend_v4_test import nlfm, stepped_frequency
from variable_bandwidth_waveform_test import clean_waveform
from waveform_classifier_stress_test import impair
from waveform_library_matcher import WaveformLibraryMatcher


FS = 40_000_000.0


LIBRARY = {
    "CW": "CW",
    "LFM": "FM",
    "NLFM": "FM",
    "STEPPED_FREQ": "FM",
    "BARKER13": "PHASE_CODED",
    "GENERIC_BIPHASE": "PHASE_CODED",
    "FRANK16_LIKE": "PHASE_CODED",
}


def clean_known_waveform(library_id, count):
    if library_id == "CW":
        return clean_waveform("CW", count, 0.30)
    if library_id == "LFM":
        return clean_waveform("LFM", count, 0.25)
    if library_id == "NLFM":
        return nlfm(count, 0.28)
    if library_id == "STEPPED_FREQ":
        return stepped_frequency(count, 7, 0.045)
    if library_id == "BARKER13":
        return clean_waveform("BARKER13", count, 0.30)
    if library_id == "GENERIC_BIPHASE":
        return clean_waveform("GENERIC_BIPHASE", count, 0.30)
    if library_id == "FRANK16_LIKE":
        return clean_waveform("FRANK16_LIKE", count, 0.30)
    raise ValueError(library_id)


def build_known_waveform_library(sample_rate_hz=FS, seed=770000):
    """Build the current experimental known-waveform acceptance library.

    This is deliberately synthetic for now. When real Ettus data is available,
    captured representative pulses should augment or replace these exemplars.
    """
    matcher = WaveformLibraryMatcher(sample_rate_hz)
    samples = []
    ids = []
    families = []
    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (10, 15, 20, 30)
    trials = 10

    for library_id, family in LIBRARY.items():
        for width_us in widths_us:
            count = int(round(width_us * 1e-6 * sample_rate_hz))
            for snr_db in snrs_db:
                for _ in range(trials):
                    seed += 1
                    rng = np.random.default_rng(seed)
                    clean = clean_known_waveform(library_id, count)
                    observed = impair(
                        clean,
                        snr_db=snr_db,
                        frequency_offset_hz=rng.uniform(-5e6, 5e6),
                        phase_rad=rng.uniform(-np.pi, np.pi),
                        timing_shift=int(rng.integers(-3, 4)),
                        seed=seed,
                    )
                    samples.append(observed)
                    ids.append(library_id)
                    families.append(family)

    matcher.fit(samples, ids, families)
    return matcher
