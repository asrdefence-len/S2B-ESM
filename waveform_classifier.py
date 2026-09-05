from dataclasses import dataclass

import numpy as np

from waveform_library import candidate_library


@dataclass(frozen=True)
class WaveformClassification:
    family: str
    subtype: str
    confidence: float
    score: float
    runner_up_family: str
    runner_up_subtype: str
    runner_up_score: float


class WaveformClassifier:
    """Lightweight intra-pulse candidate-library classifier.

    This is an experimental similarity classifier, not a calibrated emitter
    identification system. It removes amplitude, common phase and linear phase
    (carrier offset), then compares the residual complex phase structure with a
    small candidate waveform library. Both forward and conjugated candidates
    are considered so chirp/phase sign does not dominate the family decision.
    """

    def __init__(self, min_confidence=0.55):
        self.min_confidence = min_confidence

    @staticmethod
    def _normalise(samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 8:
            return None

        rms = np.sqrt(np.mean(np.abs(x) ** 2))
        if rms <= 1e-12:
            return None
        x = x / rms

        # Remove arbitrary common phase and carrier-frequency offset. Preserve
        # nonlinear phase structure: chirp and phase-code transitions.
        phase = np.unwrap(np.angle(x))
        n = np.arange(len(x), dtype=float)
        slope, intercept = np.polyfit(n, phase, 1)
        x = x * np.exp(-1j * (slope * n + intercept))
        return x

    @staticmethod
    def _complex_similarity(observed, candidate):
        candidate = WaveformClassifier._normalise(candidate)
        if candidate is None:
            return 0.0

        # Unknown absolute pulse timing can shift chip edges slightly. Test a
        # very small local shift without making this an expensive correlator.
        best = 0.0
        for shift in (-2, -1, 0, 1, 2):
            y = np.roll(candidate, shift)
            direct = abs(np.vdot(y, observed)) / len(observed)
            conjugate = abs(np.vdot(np.conj(y), observed)) / len(observed)
            best = max(best, direct, conjugate)
        return float(np.clip(best, 0.0, 1.0))

    def classify(self, pulse_iq):
        observed = self._normalise(pulse_iq)
        if observed is None:
            return WaveformClassification(
                "UNKNOWN", "UNKNOWN", 0.0, 0.0, "UNKNOWN", "UNKNOWN", 0.0
            )

        scores = []
        for candidate in candidate_library(len(observed)):
            score = self._complex_similarity(observed, candidate["samples"])
            scores.append((score, candidate["family"], candidate["subtype"]))

        scores.sort(reverse=True)
        best_score, best_family, best_subtype = scores[0]
        second_score, second_family, second_subtype = scores[1]

        # Confidence combines absolute fit with separation from the runner-up.
        margin = max(0.0, best_score - second_score)
        confidence = float(np.clip(0.75 * best_score + 0.25 * margin, 0.0, 1.0))

        if confidence < self.min_confidence:
            family = "UNKNOWN"
            subtype = "UNKNOWN"
        else:
            family = best_family
            subtype = best_subtype

        return WaveformClassification(
            family=family,
            subtype=subtype,
            confidence=confidence,
            score=best_score,
            runner_up_family=second_family,
            runner_up_subtype=second_subtype,
            runner_up_score=second_score,
        )
