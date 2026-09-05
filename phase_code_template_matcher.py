from dataclasses import dataclass

import numpy as np

from waveform_library import BARKER13, BIPHASE16, FRANK16_PHASES


@dataclass(frozen=True)
class PhaseCodeTemplateMatch:
    accepted: bool
    library_id: str
    score: float
    second_score: float


TEMPLATES = {
    "BARKER13": np.where(BARKER13 > 0, 0.0, np.pi),
    "GENERIC_BIPHASE": np.where(BIPHASE16 > 0, 0.0, np.pi),
    "FRANK16_LIKE": np.asarray(FRANK16_PHASES, dtype=float),
}


class PhaseCodeTemplateMatcher:
    """Small direct matcher for the exact phase codes currently in our library.

    This deliberately avoids a learned feature space. It estimates one complex
    phasor per candidate chip, removes a smooth residual phase trend, and compares
    the resulting chip sequence with each stored code. Only a tiny chip-boundary
    offset search is allowed. This is intended as an open-set acceptance gate,
    not a general phase-code recogniser or fingerprinting system.
    """

    def __init__(self, acceptance_score=0.72, margin=0.08):
        self.acceptance_score = float(acceptance_score)
        self.margin = float(margin)

    @staticmethod
    def _chip_phasors(samples, chip_count, offset_fraction=0.0):
        x = np.asarray(samples, dtype=np.complex128)
        n = len(x)
        if n < chip_count * 2:
            return None
        boundaries = np.linspace(0.0, n, chip_count + 1) + offset_fraction * n / chip_count
        phasors = []
        for k in range(chip_count):
            lo = max(0, int(round(boundaries[k])))
            hi = min(n, int(round(boundaries[k + 1])))
            if hi - lo < 2:
                return None
            # Avoid chip edges, where timing error and filtering smear transitions.
            trim = max(0, int(round(0.12 * (hi - lo))))
            if hi - lo - 2 * trim >= 2:
                lo += trim
                hi -= trim
            z = np.mean(x[lo:hi])
            if abs(z) < 1e-12:
                return None
            phasors.append(z / abs(z))
        return np.asarray(phasors)

    @staticmethod
    def _score(phasors, template_phase):
        observed = np.unwrap(np.angle(phasors))
        template = np.unwrap(np.asarray(template_phase, dtype=float))
        k = np.arange(len(observed), dtype=float)

        # Remove only constant + linear phase nuisance (carrier offset / phase).
        # Do not remove quadratic/cubic structure: a hybrid FM+phase signal should
        # fail an exact phase-code template match rather than be explained away.
        residual = observed - template
        coeff = np.polyfit(k, residual, 1)
        residual -= np.polyval(coeff, k)
        return float(abs(np.mean(np.exp(1j * residual))))

    def match(self, samples):
        ranked = []
        # Small boundary search only; enough for detector/sample alignment without
        # turning this into an expensive code-search engine.
        offsets = (-0.12, -0.06, 0.0, 0.06, 0.12)
        for library_id, template in TEMPLATES.items():
            best = 0.0
            for offset in offsets:
                phasors = self._chip_phasors(samples, len(template), offset)
                if phasors is not None:
                    best = max(best, self._score(phasors, template))
            ranked.append((best, library_id))

        ranked.sort(reverse=True)
        best_score, best_id = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        accepted = best_score >= self.acceptance_score and (
            best_score - second_score >= self.margin or best_score >= 0.90
        )
        return PhaseCodeTemplateMatch(
            accepted=bool(accepted),
            library_id=best_id,
            score=float(best_score),
            second_score=float(second_score),
        )
