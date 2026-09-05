import numpy as np

from phase_code_template_matcher import PhaseCodeTemplateMatcher
from waveform_known_library import FS, LIBRARY, clean_known_waveform
from waveform_classifier_stress_test import impair
from waveform_library_matcher import WaveformLibraryMatcher


class FamilySpecificWaveformLibrary:
    """Known-waveform library with direct template matching for phase codes."""

    def __init__(self, sample_rate_hz=FS):
        self.sample_rate_hz = float(sample_rate_hz)
        self.generic = WaveformLibraryMatcher(sample_rate_hz)
        self.phase = PhaseCodeTemplateMatcher()

    def match(self, samples, broad_family):
        if broad_family == "PHASE_CODED":
            return self.phase.match(samples)
        return self.generic.match(samples, broad_family=broad_family)


def build_family_specific_waveform_library(sample_rate_hz=FS, seed=880000):
    library = FamilySpecificWaveformLibrary(sample_rate_hz)

    # CW/FM still use an empirical library because their nuisance parameters are
    # naturally continuous. Phase-coded waveforms no longer train a PCA model:
    # they are compared directly with the exact stored code templates.
    generic_samples = []
    generic_ids = []
    generic_families = []

    widths_us = (2.5, 4.5, 7.0, 9.0)
    snrs_db = (5, 10, 15, 20, 30)
    trials = 12

    for library_id, family in LIBRARY.items():
        if family == "PHASE_CODED":
            continue
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
                    generic_samples.append(observed)
                    generic_ids.append(library_id)
                    generic_families.append(family)

    library.generic.fit(generic_samples, generic_ids, generic_families)
    return library
