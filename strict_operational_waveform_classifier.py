from dataclasses import dataclass

from operational_waveform_classifier_cyclic import CyclicOperationalWaveformClassifier
from waveform_known_library import build_known_waveform_library


@dataclass(frozen=True)
class StrictWaveformClassification:
    family: str
    confidence: float
    library_id: str
    library_accepted: bool
    library_distance_ratio: float
    broad_family: str
    broad_scores: dict
    rejection_reason: str


class StrictOperationalWaveformClassifier:
    """Broad physics classification followed by strict known-library acceptance."""

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.broad = CyclicOperationalWaveformClassifier(sample_rate_hz)
        self.library = build_known_waveform_library(sample_rate_hz)

    def classify(self, samples):
        broad = self.broad.classify(samples)
        if broad.family == "UNKNOWN":
            return StrictWaveformClassification(
                family="UNKNOWN",
                confidence=broad.confidence,
                library_id="NONE",
                library_accepted=False,
                library_distance_ratio=float("inf"),
                broad_family="UNKNOWN",
                broad_scores=broad.scores,
                rejection_reason=broad.rejection_reason or "broad-family classifier rejected waveform",
            )

        match = self.library.match(samples, broad_family=broad.family)
        if not match.accepted:
            return StrictWaveformClassification(
                family="UNKNOWN",
                confidence=broad.confidence,
                library_id=match.library_id,
                library_accepted=False,
                library_distance_ratio=match.distance_ratio,
                broad_family=broad.family,
                broad_scores=broad.scores,
                rejection_reason="outside known-waveform library acceptance region",
            )

        # Confidence remains an engineering evidence value, not a calibrated
        # posterior probability. Penalise broad confidence as library distance
        # approaches the empirical acceptance boundary.
        library_quality = max(0.0, 1.0 - 0.5 * match.distance_ratio)
        confidence = min(float(broad.confidence), float(library_quality))
        return StrictWaveformClassification(
            family=broad.family,
            confidence=confidence,
            library_id=match.library_id,
            library_accepted=True,
            library_distance_ratio=match.distance_ratio,
            broad_family=broad.family,
            broad_scores=broad.scores,
            rejection_reason="",
        )
