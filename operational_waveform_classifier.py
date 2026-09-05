from dataclasses import dataclass

import numpy as np

from physics_waveform_frontend_v6 import PhysicsWaveformFrontendV6


@dataclass(frozen=True)
class OperationalWaveformClassification:
    family: str
    confidence: float
    scores: dict
    phase_rms_linear: float
    phase_rms_quadratic: float
    phase_rms_cubic: float
    phase_jump_strength: float
    frequency_excursion_hz: float


class OperationalWaveformClassifier:
    """Broad waveform classifier for the current S2B ESM experiment.

    Operational classes are intentionally limited to:
      CW
      FM              (LFM, NLFM, stepped/frequency-coded, other frequency variation)
      PHASE_CODED
      UNKNOWN

    Exact waveform fingerprinting is deferred. The underlying phase and frequency
    observables are retained so later S2B change detection can still see waveform
    changes inside a broad family.
    """

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.frontend = PhysicsWaveformFrontendV6(sample_rate_hz)

    def classify(self, samples):
        r = self.frontend.classify(samples)

        # Collapse all frequency-varying classes into FM for the present ESM.
        fm_score = max(
            float(r.scores.get("FM", 0.0)),
            float(r.scores.get("FREQUENCY_CODED", 0.0)),
        )
        scores = {
            "CW": float(r.scores.get("CW", 0.0)),
            "FM": fm_score,
            "PHASE_CODED": float(r.scores.get("PHASE_CODED", 0.0)),
        }

        family = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0.0
        margin = best - second

        # Keep UNKNOWN conservative: use it when no supported broad family gives
        # adequate evidence or when all evidence remains weakly ambiguous.
        if best < 0.20 or (best < 0.38 and margin < 0.04):
            family = "UNKNOWN"

        confidence = float(np.clip(best, 0.0, 1.0))
        return OperationalWaveformClassification(
            family=family,
            confidence=confidence,
            scores=scores,
            phase_rms_linear=r.phase_rms_linear,
            phase_rms_quadratic=r.phase_rms_quadratic,
            phase_rms_cubic=r.phase_rms_cubic,
            phase_jump_strength=r.phase_jump_strength,
            frequency_excursion_hz=r.local_frequency_excursion_hz,
        )
