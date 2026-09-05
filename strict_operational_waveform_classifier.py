from dataclasses import dataclass

from family_specific_waveform_library import build_family_specific_waveform_library
from operational_waveform_classifier_cyclic import CyclicOperationalWaveformClassifier


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
    """Broad physics classification followed by family-specific library acceptance."""

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.broad = CyclicOperationalWaveformClassifier(sample_rate_hz)
        self.library = build_family_specific_waveform_library(sample_rate_hz)

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

        match = self.library.match(samples, broad.family)
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
