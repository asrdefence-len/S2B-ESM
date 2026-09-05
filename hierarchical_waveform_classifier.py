from dataclasses import dataclass

from bandwidth_normalized_fft_pca_classifier import BandwidthNormalizedFFTPCAClassifier
from physics_waveform_frontend_v3 import PhysicsWaveformFrontendV3


@dataclass(frozen=True)
class HierarchicalWaveformClassification:
    family: str
    subtype: str
    confidence: float
    coarse_family: str
    occupied_bandwidth_hz: float
    family_scores: dict
    physics_scores: dict


class HierarchicalWaveformClassifier:
    """Physics-first experimental waveform classifier.

    V3 routes smooth CW/LFM directly. Pulses routed as PHASE_CODED are passed to
    a bandwidth-normalized FFT/PCA model trained only to distinguish BIPHASE from
    POLYPHASE. Absolute occupied bandwidth remains a separate observable.

    Subtype identification is deliberately not claimed yet. Barker correlation and
    polyphase-code correlation belong in later optional refinement stages.
    """

    def __init__(self, sample_rate_hz=40_000_000.0, feature_bins=256, n_components=8):
        self.sample_rate_hz = float(sample_rate_hz)
        self.physics = PhysicsWaveformFrontendV3(sample_rate_hz)
        self.coded = BandwidthNormalizedFFTPCAClassifier(
            sample_rate_hz=sample_rate_hz,
            feature_bins=feature_bins,
            n_components=n_components,
        )
        self._coded_fitted = False

    def fit_coded(self, samples, labels):
        allowed = {"BIPHASE", "POLYPHASE"}
        filtered_samples = []
        filtered_labels = []
        for samples_i, label in zip(samples, labels):
            if label in allowed:
                filtered_samples.append(samples_i)
                filtered_labels.append(label)
        if not filtered_samples:
            raise ValueError("No BIPHASE/POLYPHASE training examples")
        self.coded.fit(filtered_samples, filtered_labels)
        self._coded_fitted = True
        return self

    def classify(self, samples):
        physics = self.physics.classify(samples)
        physics_scores = {
            "CW": physics.cw_score,
            "LFM": physics.lfm_score,
            "PHASE_CODED": physics.phase_coded_score,
        }

        # Always use the spectral extractor for occupied bandwidth, even when the
        # physics router makes the family decision.
        _, occupied_bw = self.coded._spectral_feature(samples)

        if physics.coarse_family == "CW":
            return HierarchicalWaveformClassification(
                family="CW",
                subtype="CW",
                confidence=physics.cw_score,
                coarse_family="CW",
                occupied_bandwidth_hz=occupied_bw,
                family_scores={"CW": physics.cw_score},
                physics_scores=physics_scores,
            )

        if physics.coarse_family == "LFM":
            return HierarchicalWaveformClassification(
                family="LFM",
                subtype="LFM",
                confidence=physics.lfm_score,
                coarse_family="LFM",
                occupied_bandwidth_hz=occupied_bw,
                family_scores={"LFM": physics.lfm_score},
                physics_scores=physics_scores,
            )

        if physics.coarse_family == "PHASE_CODED":
            if not self._coded_fitted:
                raise RuntimeError("Coded FFT/PCA stage must be fit before classifying PHASE_CODED pulses")
            coded = self.coded.classify(samples)
            return HierarchicalWaveformClassification(
                family=coded.family,
                subtype="CODED_UNKNOWN",
                confidence=coded.confidence,
                coarse_family="PHASE_CODED",
                occupied_bandwidth_hz=coded.occupied_bandwidth_hz,
                family_scores=coded.family_scores,
                physics_scores=physics_scores,
            )

        return HierarchicalWaveformClassification(
            family="UNKNOWN",
            subtype="UNKNOWN",
            confidence=max(physics_scores.values()),
            coarse_family="UNKNOWN",
            occupied_bandwidth_hz=occupied_bw,
            family_scores={},
            physics_scores=physics_scores,
        )
