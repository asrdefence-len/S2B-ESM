from dataclasses import dataclass

import numpy as np

from hierarchical_waveform_classifier import HierarchicalWaveformClassifier


@dataclass(frozen=True)
class OpenSetWaveformClassification:
    family: str
    candidate_family: str
    confidence: float
    accepted: bool
    rejection_reason: str
    occupied_bandwidth_hz: float
    family_scores: dict
    physics_scores: dict
    coded_distance_ratio: float


class OpenSetWaveformClassifier:
    """Open-set wrapper around the hard hierarchical classifier.

    Known families may be rejected as UNKNOWN when evidence is weak or, for coded
    waveforms, when the PCA representation lies too far from the learned family
    centroid. Thresholds are engineering starting points and must be validated on
    unseen waveforms rather than treated as calibrated probabilities.
    """

    def __init__(self, sample_rate_hz=40_000_000.0, min_physics_score=0.18,
                 min_coded_confidence=0.18, coded_distance_ratio_limit=1.35):
        self.base = HierarchicalWaveformClassifier(sample_rate_hz)
        self.min_physics_score = float(min_physics_score)
        self.min_coded_confidence = float(min_coded_confidence)
        self.coded_distance_ratio_limit = float(coded_distance_ratio_limit)
        self._coded_radius = {}

    def fit_coded(self, samples, labels):
        self.base.fit_coded(samples, labels)
        distances = {"BIPHASE": [], "POLYPHASE": []}
        for waveform, label in zip(samples, labels):
            if label not in distances:
                continue
            feature, _ = self.base.coded._spectral_feature(waveform)
            if feature is None:
                continue
            z = self.base.coded._project_matrix(feature[None, :])[0]
            centroid = self.base.coded.centroids_[label]
            distances[label].append(float(np.linalg.norm(z - centroid)))
        for label, values in distances.items():
            if values:
                # Robust high-percentile training envelope. The multiplier gives
                # modest allowance for test impairments not exactly seen in fit.
                self._coded_radius[label] = 1.15 * float(np.quantile(values, 0.95))
        return self

    def classify(self, samples):
        result = self.base.classify(samples)
        candidate = result.family
        reason = ""
        accepted = True
        distance_ratio = 0.0

        if candidate in ("CW", "LFM"):
            score = result.physics_scores.get(candidate, 0.0)
            if score < self.min_physics_score:
                accepted = False
                reason = "weak physics evidence"

        elif candidate in ("BIPHASE", "POLYPHASE"):
            if result.confidence < self.min_coded_confidence:
                accepted = False
                reason = "weak coded-family separation"
            feature, _ = self.base.coded._spectral_feature(samples)
            if feature is not None and candidate in self._coded_radius:
                z = self.base.coded._project_matrix(feature[None, :])[0]
                distance = float(np.linalg.norm(z - self.base.coded.centroids_[candidate]))
                radius = max(self._coded_radius[candidate], 1e-9)
                distance_ratio = distance / radius
                if distance_ratio > self.coded_distance_ratio_limit:
                    accepted = False
                    reason = "outside learned coded-family envelope"
        else:
            accepted = False
            reason = "no supported family"

        family = candidate if accepted else "UNKNOWN"
        return OpenSetWaveformClassification(
            family=family,
            candidate_family=candidate,
            confidence=result.confidence,
            accepted=accepted,
            rejection_reason=reason,
            occupied_bandwidth_hz=result.occupied_bandwidth_hz,
            family_scores=result.family_scores,
            physics_scores=result.physics_scores,
            coded_distance_ratio=distance_ratio,
        )
