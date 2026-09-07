"""Conditional lightweight-cyclic operational waveform classifier.

This is an experimental drop-in candidate for the current cyclic classifier.
The envelope spectrum is always evaluated because it provides the independent
multi-component rejection gate. x^2/x^4 spectra are evaluated only when the
physics path already makes PHASE_CODED plausible, which is the only circumstance
in which the production classifier uses those features to change scores.
"""

from dataclasses import dataclass

import numpy as np

from cyclostationary_waveform_diagnostic import spectral_concentration
from operational_waveform_classifier import OperationalWaveformClassifier


@dataclass(frozen=True)
class FastCyclicOperationalWaveformClassification:
    family: str
    confidence: float
    scores: dict
    base_family: str
    cyclic_features: dict
    rejection_reason: str


class FastCyclicOperationalWaveformClassifier:
    def __init__(self, sample_rate_hz=40_000_000.0):
        self.base = OperationalWaveformClassifier(sample_rate_hz)

    @staticmethod
    def _empty_cyclic_features():
        return {
            "x2_peak": 0.0, "x2_top4": 0.0, "x2_entropy": 0.0,
            "x4_peak": 0.0, "x4_top4": 0.0, "x4_entropy": 0.0,
            "env_peak": 0.0, "env_top4": 0.0, "env_entropy": 0.0,
            "x2_x4_evaluated": False,
        }

    def classify(self, samples):
        base = self.base.classify(samples)
        scores = dict(base.scores)
        reason = ""
        x = np.asarray(samples, dtype=np.complex128)
        cf = self._empty_cyclic_features()

        # The production rejection rule depends only on envelope cyclic evidence,
        # so evaluate that first for every pulse.
        pe, te, ee = spectral_concentration(np.abs(x) ** 2)
        cf["env_peak"] = pe
        cf["env_top4"] = te
        cf["env_entropy"] = ee

        envelope_multicomponent = te > 0.18 and pe > 0.045 and ee < 0.86
        if envelope_multicomponent:
            return FastCyclicOperationalWaveformClassification(
                family="UNKNOWN",
                confidence=float(np.clip(max(scores.values()), 0.0, 1.0)),
                scores=scores,
                base_family=base.family,
                cyclic_features=cf,
                rejection_reason="strong cyclic envelope beating / multi-component structure",
            )

        # x^2/x^4 evidence is used by the original classifier only to corroborate
        # an already plausible PHASE_CODED interpretation. Skip both FFTs for the
        # common obvious CW/FM case without changing the decision equations.
        phase_plausible = base.family == "PHASE_CODED" or scores.get("PHASE_CODED", 0.0) >= 0.35
        if phase_plausible:
            p2, t2, e2 = spectral_concentration(x ** 2)
            p4, t4, e4 = spectral_concentration(x ** 4)
            cf.update({
                "x2_peak": p2, "x2_top4": t2, "x2_entropy": e2,
                "x4_peak": p4, "x4_top4": t4, "x4_entropy": e4,
                "x2_x4_evaluated": True,
            })
            biphase_support = np.clip((t2 - 0.18) / 0.45, 0.0, 1.0)
            polyphase_support = np.clip((t4 - 0.12) / 0.45, 0.0, 1.0)
            cyclic_support = float(max(biphase_support, polyphase_support))
            scores["PHASE_CODED"] = float(np.clip(
                0.80 * scores.get("PHASE_CODED", 0.0) + 0.20 * cyclic_support,
                0.0, 1.0,
            ))

        family = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0.0
        if best < 0.20 or (best < 0.38 and best - second < 0.04):
            family = "UNKNOWN"
            reason = "weak or ambiguous broad-family evidence"

        return FastCyclicOperationalWaveformClassification(
            family=family,
            confidence=float(np.clip(best, 0.0, 1.0)),
            scores=scores,
            base_family=base.family,
            cyclic_features=cf,
            rejection_reason=reason,
        )
