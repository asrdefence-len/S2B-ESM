import numpy as np

from phase_code_library_matcher import PhaseCodeLibraryMatcher
from waveform_known_library import FS, LIBRARY, clean_known_waveform
from waveform_classifier_stress_test import impair
from waveform_library_matcher import WaveformLibraryMatcher


class FamilySpecificWaveformLibrary:
    """Known-waveform library with a phase-code-specific fingerprint path."""

    def __init__(self, sample_rate_hz=FS):
        self.sample_rate_hz = float(sample_rate_hz)
        self.generic = WaveformLibraryMatcher(sample_rate_hz)
        self.phase = PhaseCodeLibraryMatcher()

    def match(self, samples, broad_family):
        if broad_family == "PHASE_CODED":
            return self.phase.match(samples)
        return self.generic.match(samples, broad_family=broad_family)


def build_family_specific_waveform_library(sample_rate_hz=FS, seed=880000):
    library = FamilySpecificWaveformLibrary(sample_rate_hz)

    generic_samples = []
    generic_ids = []
    generic_families = []
    phase_samples = []
    phase_ids = []

    widths_us = (2.5, 4.5, 7.0, 9.0)
    # Include 5 dB in the training envelope because the previous strict matcher
    # rejected too many legitimate CW/FM observations there.
    snrs_db = (5, 10, 15, 20, 30)
    trials = 12

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
                    if family == "PHASE_CODED":
                        phase_samples.append(observed)
                        phase_ids.append(library_id)
                    else:
                        generic_samples.append(observed)
                        generic_ids.append(library_id)
                        generic_families.append(family)

    library.generic.fit(generic_samples, generic_ids, generic_families)
    library.phase.fit(phase_samples, phase_ids)
    return library
