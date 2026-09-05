from dataclasses import dataclass

from bandwidth_normalized_fft_pca_classifier import BandwidthNormalizedFFTPCAClassifier
from physics_waveform_frontend_v3 import PhysicsWaveformFrontendV3


@dataclass(frozen=True)
class SoftRoutedWaveformClassification:
    family: str
    confidence: float
    occupied_bandwidth_hz: float
    family_scores: dict
    physics_scores: dict
    coded_stage_run: bool


class SoftRoutedWaveformClassifier:
    """Soft-gated physics + coded-spectrum waveform classifier.

    Clear CW/LFM pulses stay on the cheap physics path. The coded PCA stage is
    also evaluated whenever phase-coded evidence is competitive with the best
    smooth-waveform evidence. Final family evidence combines the two stages rather
    than treating the first-stage route as irreversible.
    """

    def __init__(self, sample_rate_hz=40_000_000.0, feature_bins=256, n_components=8,
                 coded_gate_ratio=0.55, coded_gate_floor=0.05):
        self.sample_rate_hz = float(sample_rate_hz)
        self.coded_gate_ratio = float(coded_gate_ratio)
        self.coded_gate_floor = float(coded_gate_floor)
        self.physics = PhysicsWaveformFrontendV3(sample_rate_hz)
        self.coded = BandwidthNormalizedFFTPCAClassifier(
            sample_rate_hz=sample_rate_hz,
            feature_bins=feature_bins,
            n_components=n_components,
        )
        self._coded_fitted = False

    def fit_coded(self, samples, labels):
        kept_samples = []
        kept_labels = []
        for waveform, label in zip(samples, labels):
            if label in ("BIPHASE", "POLYPHASE"):
                kept_samples.append(waveform)
                kept_labels.append(label)
        if not kept_samples:
            raise ValueError("No BIPHASE/POLYPHASE training examples")
        self.coded.fit(kept_samples, kept_labels)
        self._coded_fitted = True
        return self

    @staticmethod
    def _normalise(scores):
        total = sum(max(float(v), 0.0) for v in scores.values())
        if total <= 1e-12:
            return {key: 0.0 for key in scores}
        return {key: max(float(value), 0.0) / total for key, value in scores.items()}

    def classify(self, samples):
        physics = self.physics.classify(samples)
        physics_scores = {
            "CW": physics.cw_score,
            "LFM": physics.lfm_score,
            "PHASE_CODED": physics.phase_coded_score,
        }
        _, occupied_bw = self.coded._spectral_feature(samples)

        smooth_best = max(physics.cw_score, physics.lfm_score)
        run_coded = (
            physics.coarse_family == "PHASE_CODED"
            or physics.phase_coded_score >= self.coded_gate_floor
            or physics.phase_coded_score >= self.coded_gate_ratio * max(smooth_best, 1e-9)
        )

        raw_scores = {
            "CW": physics.cw_score,
            "LFM": physics.lfm_score,
            "BIPHASE": 0.0,
            "POLYPHASE": 0.0,
        }

        if run_coded:
            if not self._coded_fitted:
                raise RuntimeError("Coded FFT/PCA stage must be fit before soft routing")
            coded = self.coded.classify(samples)
            occupied_bw = coded.occupied_bandwidth_hz
            # The physics phase-coded score is the gate/evidence that this branch is
            # relevant; PCA then distributes that evidence between coded families.
            raw_scores["BIPHASE"] = physics.phase_coded_score * coded.family_scores.get("BIPHASE", 0.0)
            raw_scores["POLYPHASE"] = physics.phase_coded_score * coded.family_scores.get("POLYPHASE", 0.0)

        family_scores = self._normalise(raw_scores)
        family = max(family_scores, key=family_scores.get)
        confidence = family_scores[family]
        if confidence <= 0.0:
            family = "UNKNOWN"

        return SoftRoutedWaveformClassification(
            family=family,
            confidence=confidence,
            occupied_bandwidth_hz=occupied_bw,
            family_scores=family_scores,
            physics_scores=physics_scores,
            coded_stage_run=run_coded,
        )
