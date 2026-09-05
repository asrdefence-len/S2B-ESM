from dataclasses import dataclass

import numpy as np

from cyclostationary_waveform_diagnostic import cyclic_features
from operational_waveform_classifier import OperationalWaveformClassifier


@dataclass(frozen=True)
class CyclicOperationalWaveformClassification:
    family: str
    confidence: float
    scores: dict
    base_family: str
    cyclic_features: dict
    rejection_reason: str


class CyclicOperationalWaveformClassifier:
    """Operational CW/FM/PHASE_CODED/UNKNOWN classifier with lightweight cyclic evidence.

    The base physics classifier remains primary. The cyclic features are used only
    to reject obvious envelope-beating/multi-component cases and to corroborate
    phase-coded evidence. This is intentionally not a full SCF implementation.
    """

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.base = OperationalWaveformClassifier(sample_rate_hz)

    def classify(self, samples):
        base = self.base.classify(samples)
        cf = cyclic_features(samples)
        scores = dict(base.scores)
        reason = ""

        # Strong envelope periodicity is a reliable signature of simultaneous
        # multi-component beating in our current stress case. Normal radar-like
        # waveforms had env_top4 ~0.04 and high envelope entropy; dual tone was far
        # more concentrated. Use a deliberately conservative rejection gate.
        envelope_multicomponent = (
            cf["env_top4"] > 0.18
            and cf["env_peak"] > 0.045
            and cf["env_entropy"] < 0.86
        )

        if envelope_multicomponent:
            return CyclicOperationalWaveformClassification(
                family="UNKNOWN",
                confidence=float(np.clip(max(scores.values()), 0.0, 1.0)),
                scores=scores,
                base_family=base.family,
                cyclic_features=cf,
                rejection_reason="strong cyclic envelope beating / multi-component structure",
            )

        # Corroboration only: biphase tends to concentrate strongly after squaring,
        # while polyphase often concentrates after fourth power. Do not allow these
        # features by themselves to turn CW/FM into PHASE_CODED; only strengthen an
        # already plausible phase-coded interpretation from the physics path.
        if base.family == "PHASE_CODED" or scores.get("PHASE_CODED", 0.0) >= 0.35:
            biphase_support = np.clip((cf["x2_top4"] - 0.18) / 0.45, 0.0, 1.0)
            polyphase_support = np.clip((cf["x4_top4"] - 0.12) / 0.45, 0.0, 1.0)
            cyclic_support = float(max(biphase_support, polyphase_support))
            scores["PHASE_CODED"] = float(np.clip(
                0.80 * scores.get("PHASE_CODED", 0.0) + 0.20 * cyclic_support,
                0.0,
                1.0,
            ))

        family = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0.0
        if best < 0.20 or (best < 0.38 and best - second < 0.04):
            family = "UNKNOWN"
            reason = "weak or ambiguous broad-family evidence"

        return CyclicOperationalWaveformClassification(
            family=family,
            confidence=float(np.clip(best, 0.0, 1.0)),
            scores=scores,
            base_family=base.family,
            cyclic_features=cf,
            rejection_reason=reason,
        )
