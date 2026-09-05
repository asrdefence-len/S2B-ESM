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

    The classifier uses differential phase, x[n] * conj(x[n-1]), as its main
    representation. This removes arbitrary common carrier phase and turns a
    constant carrier-frequency offset into an approximately constant phase
    increment. Removing that mean increment leaves the phase transitions that
    distinguish biphase and polyphase coding, while LFM retains a changing
    phase increment.

    This is an experimental engineering similarity classifier. Its confidence
    is not a calibrated probability.
    """

    def __init__(self, min_confidence=0.45):
        self.min_confidence = min_confidence

    @staticmethod
    def _normalise(samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 8:
            return None
        rms = np.sqrt(np.mean(np.abs(x) ** 2))
        if rms <= 1e-12:
            return None
        return x / rms

    @staticmethod
    def _differential_phase_signature(samples):
        x = WaveformClassifier._normalise(samples)
        if x is None:
            return None

        d = x[1:] * np.conj(x[:-1])
        magnitude = np.abs(d)
        valid = magnitude > 1e-8
        if not np.any(valid):
            return None
        d = d / np.maximum(magnitude, 1e-12)

        # Estimate the nuisance carrier increment robustly. Most samples in a
        # phase-coded pulse lie inside chips, so the circular mean is dominated
        # by the carrier rather than by the relatively sparse chip transitions.
        carrier_increment = np.angle(np.sum(d[valid]))
        d = d * np.exp(-1j * carrier_increment)
        return d

    @staticmethod
    def _signature_similarity(observed, candidate):
        reference = WaveformClassifier._differential_phase_signature(candidate)
        if reference is None or len(reference) != len(observed):
            return 0.0

        # Compare the differential-phase sequences directly. A small timing
        # shift handles sample/chip-boundary uncertainty. Conjugation permits
        # opposite phase/chirp sense without changing the waveform family.
        best = 0.0
        for shift in (-2, -1, 0, 1, 2):
            y = np.roll(reference, shift)
            direct = abs(np.vdot(y, observed)) / len(observed)
            conjugate = abs(np.vdot(np.conj(y), observed)) / len(observed)
            best = max(best, direct, conjugate)
        return float(np.clip(best, 0.0, 1.0))

    @staticmethod
    def _transition_similarity(observed, candidate):
        """Emphasise samples where phase changes relative to the carrier.

        Raw differential-phase correlation is dominated by within-chip samples
        near zero residual phase. Weighting transition samples makes Barker,
        generic biphase and polyphase structure much more discriminating.
        """
        reference = WaveformClassifier._differential_phase_signature(candidate)
        if reference is None or len(reference) != len(observed):
            return 0.0

        obs_phase = np.angle(observed)
        best = 0.0
        for shift in (-2, -1, 0, 1, 2):
            ref = np.roll(reference, shift)
            ref_phase = np.angle(ref)
            weight = 0.10 + np.maximum(np.abs(ref_phase), np.abs(obs_phase)) / np.pi
            direct_error = np.angle(observed * np.conj(ref))
            conjugate_error = np.angle(observed * ref)
            direct = np.sum(weight * np.cos(direct_error)) / np.sum(weight)
            conjugate = np.sum(weight * np.cos(conjugate_error)) / np.sum(weight)
            score = 0.5 * (1.0 + max(direct, conjugate))
            best = max(best, score)
        return float(np.clip(best, 0.0, 1.0))

    def classify(self, pulse_iq):
        observed = self._differential_phase_signature(pulse_iq)
        if observed is None:
            return WaveformClassification(
                "UNKNOWN", "UNKNOWN", 0.0, 0.0, "UNKNOWN", "UNKNOWN", 0.0
            )

        scores = []
        for candidate in candidate_library(len(pulse_iq)):
            broad = self._signature_similarity(observed, candidate["samples"])
            transitions = self._transition_similarity(observed, candidate["samples"])
            score = 0.35 * broad + 0.65 * transitions
            scores.append((score, candidate["family"], candidate["subtype"]))

        scores.sort(reverse=True)
        best_score, best_family, best_subtype = scores[0]
        second_score, second_family, second_subtype = scores[1]

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
